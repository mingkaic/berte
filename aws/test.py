# standard packages
import time
import logging
import yaml
import tarfile
import os.path
from urllib.request import Request, urlopen

# installed packages
import tensorflow_text as text
import tensorflow as tf

# local packages
import model
import dataset
import traintest
import telemetry
import common.training as training

# aws packages
from cloudwatch import cloudwatch
import boto3

from aws_common.instance import get_instance

# --------- AWS Parameters ---------
_instance_info = get_instance()
INSTANCE_ID = _instance_info['instance_id']
EC2_REGION = _instance_info['ec2_region']

S3_BUCKET = 'bidi-enc-rep-trnsformers-everywhere'
S3_DIR = 'v1/pretraining'
CLOUDWATCH_GROUP = 'bidi-enc-rep-trnsformers-everywhere'
ID = 'test'

s3_client = boto3.client('s3')
s3_configs_key = os.path.join(S3_DIR, ID, 's3_configs.tar.gz')
s3_export_key = os.path.join(S3_DIR, ID, 's3_export.tar.gz')
configs_resp = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_configs_key)
export_resp = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_export_key)

with open('s3_configs.tar.gz', 'wb') as f:
    f.write(configs_resp['Body'].read())
configs_tar = tarfile.open("s3_configs.tar.gz")
configs_tar.extractall()
configs_tar.close()

with open('s3_export.tar.gz', 'wb') as f:
    f.write(export_resp['Body'].read())
export_tar = tarfile.open("s3_export.tar.gz")
export_tar.extractall()
export_tar.close()

# --------- Setup from configs ---------
logger=logging.getLogger(ID)
formatter = logging.Formatter('%(asctime)s : %(levelname)s - %(message)s')
handler = cloudwatch.CloudwatchHandler(log_group = CLOUDWATCH_GROUP, region=EC2_REGION)

handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

with open("configs/pretrain_1.yaml") as f:
    _args = yaml.safe_load(f.read())
    tokenizer_filename = _args["tokenizer_model"]
    training_args = _args["training_args"]
    model_args = _args["model_args"]
    training_settings = _args["training_settings"]

with open("configs/ps_dataset_1.yaml") as f:
    _args = yaml.safe_load(f.read())
    dataset_path = _args["dataset"]
    dataset_shard_dirpath = _args["shard_directory"]
    tokenizer_setup = _args["tokenizer_args"]

with open(tokenizer_filename, 'rb') as f:
    tokenizer = text.SentencepieceTokenizer(model=f.read(),
                                            out_type=tf.int32,
                                            add_bos=tokenizer_setup["add_bos"],
                                            add_eos=tokenizer_setup["add_eos"])

# generate or retrieve cached values
cached_args = dataset.cache_values("configs/pretrain_cache.yaml", {
    "dataset_width": dataset.generate_dataset_width,
    "vocab_size": lambda _, tokenizer: int(tokenizer.vocab_size().numpy()),
}, dataset_path, tokenizer)

# --------- Arguments ---------
training_shards, _ = dataset.setup_shards(dataset_shard_dirpath, training_args,
                                                      tokenizer, logger=logger)
learning_rate = training.CustomSchedule(model_args["model_dim"])
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

for training_key in training_shards:
    training_shards[training_key] = training_shards[training_key].take(1)

_builder = model.InitParamBuilder()
for param in cached_args:
    _builder.add_param(cached_args[param], param)
for param in model_args:
    _builder.add_param(model_args[param], param)
pretrainer = model.PretrainerMLM(
    tokenizer=tokenizer,
    params=_builder.build(),
    metadata=model.PretrainerMLMMetadataBuilder().
        tokenizer_meta(tokenizer_filename).
        optimizer_iter(optimizer.iterations).build())
_batch_shape = (training_args["batch_size"], cached_args["dataset_width"])
run_test = traintest.build_tester(pretrainer,
        samples=[
            "este é um problema que temos que resolver.",
            "this is a problem we have to solve .",
            "os meus vizinhos ouviram sobre esta ideia.",
            "and my neighboring homes heard about this idea .",
            "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas "
            "mágicas que aconteceram.",
            "so i 'll just share with you some stories very quickly of some magical things that "
            "have happened .",
            "este é o primeiro livro que eu fiz.",
            "this is the first book i've ever done.",
        ],
        mask_sizes=_batch_shape,
        logger=logger)
train_batch, train_loss, train_accuracy = traintest.build_pretrainer(
        pretrainer, optimizer, _batch_shape)

try:
    os.makedirs('out')
except FileExistsError:
    pass

def save_out():
    compress_file = 'out.tar.gz'
    with tarfile.open(compress_file, 'w:gz') as tar:
        tar.add('out', arcname=os.path.basename('out'))

    s3_client.upload_file(compress_file,
            Bucket=S3_BUCKET,
            Key=os.path.join(S3_DIR, ID, compress_file))

ckpt = tf.train.Checkpoint(pretrainer=pretrainer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          directory="out/checkpoints/test",
                                          max_to_keep=50)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    logger.info('Latest checkpoint restored!!')

# allow at most 50 bad batches per 100 batch, recover 10 quota afterwards
QUOTA_BUCKET_CAPACITY = 50
QUOTA_BUCKET_RECOVER = 10
QUOTA_BUCKET_RECOVER_RATE = 50
SKIP_LOSS_WINDOW = 200

bucket = training.QuotaBucket(training_settings["skip_bad_loss"]["warmup"],
                              bucket_capacity=QUOTA_BUCKET_CAPACITY,
                              bucket_recover=QUOTA_BUCKET_RECOVER,
                              bucket_recover_rate=QUOTA_BUCKET_RECOVER_RATE)
prev_losses = training.LossWindow(capacity=SKIP_LOSS_WINDOW)
nan_reporter = telemetry.tokens_reporter(logger, tokenizer)
trainer = traintest.EpochPretrainer(
    traintest.EpochPretrainerInitBuilder().
        training_settings(training_settings).
        training_loss(train_loss).
        training_accuracy(train_accuracy).
        training_shards(training_shards).
        training_cb(lambda batch, lengths, loss_check: train_batch(
            batch, lengths, loss_check,
            mask_rate=training_args["mask_rate"],
            context_rate=training_args["context_rate"])).
        ckpt_manager(ckpt_manager).
        bucket(bucket).
        prev_losses(prev_losses).build())

def run_epochs(epochs, skip_shards=None):
    """ run all epoch """
    for epoch in range(epochs):
        start = time.time()
        sublogger = telemetry.PrefixAdapter(logger, 'Epoch {}'.format(epoch+1))
        trainer.run_epoch(skip_shards, logger=sublogger, nan_reporter=nan_reporter)

        if (epoch + 1) % training_settings["epochs_per_save"] == 0:
            logger.info('Saving checkpoint for epoch %d at %s', epoch+1, ckpt_manager.save())

        if (epoch + 1) % training_settings["epochs_per_test"] == 0:
            run_test() # run every epoch

        logger.info('Epoch %d Loss %.4f Accuracy %.4f',
                epoch+1, train_loss.result(), train_accuracy.result())
        logger.info('Time taken for 1 epoch: %.2f secs', time.time() - start)

# --------- Training ---------
run_epochs(1)

save_name = 'out/test'
tf.saved_model.save(pretrainer, save_name)
save_out()

boto3.client('ec2', region_name=EC2_REGION).stop_instances(InstanceIds=[INSTANCE_ID])
