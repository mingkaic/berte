#!/usr/bin/env python3
"""
This module defines 4_persentence_pretrain_mlm_15p.ipynb logic
"""
# standard packages
import time
import logging
import os.path

import yaml

# installed packages
import tensorflow_text as text
import tensorflow as tf

# local packages
import models
import dataset
import traintest
import telemetry
import common.training as training

# allow at most 50 bad batches per 100 batch, recover 10 quota afterwards
QUOTA_BUCKET_CAPACITY = 50
QUOTA_BUCKET_RECOVER = 10
QUOTA_BUCKET_RECOVER_RATE = 50
SKIP_LOSS_WINDOW = 200

def _setup_pretrainer(tokenizer_filename, tokenizer_setup, optimizer, in_model_dir, *args):
    builder = models.InitParamBuilder()
    for arg in args:
        for key in arg:
            builder.add_param(arg[key], key)

    with open(tokenizer_filename, 'rb') as file:
        tokenizer = text.SentencepieceTokenizer(model=file.read(),
                                                out_type=tf.int32,
                                                add_bos=tokenizer_setup["add_bos"],
                                                add_eos=tokenizer_setup["add_eos"])
    metadata = models.PretrainerMLMMetadataBuilder().\
        tokenizer_meta(tokenizer_filename).\
        optimizer_iter(optimizer.iterations).build()

    if in_model_dir is not None:
        return models.PretrainerMLM.load(tokenizer, metadata, in_model_dir)

    return models.PretrainerMLM(
            tokenizer=tokenizer,
            params=builder.build(),
            metadata=metadata)

def _setup_cached_args(tokenizer_filename, tokenizer_setup, dataset_path):
    with open(tokenizer_filename, 'rb') as file:
        tokenizer = text.SentencepieceTokenizer(model=file.read(),
                                                out_type=tf.int32,
                                                add_bos=tokenizer_setup["add_bos"],
                                                add_eos=tokenizer_setup["add_eos"])
    # generate or retrieve cached values
    return dataset.cache_values("configs/pretrain_cache.yaml", {
        "dataset_width": dataset.generate_dataset_width,
        "vocab_size": lambda _, tokenizer: int(tokenizer.vocab_size().numpy()),
    }, dataset_path, tokenizer)

def main(logger, outdir,
        nepochs=1,
        in_model_dir=None,
        ckpt_id='train_15p_ps',
        model_id='berte_pretrain_mlm_15p',
        training_preprocessing=None,
        context_rate_overwrite=None,
        ckpt_options=None):
    """ actually pretrains some model """

    if ckpt_options is None:
        ckpt_options = tf.train.CheckpointOptions()

    # --------- Extract configs and cached values ---------
    with open("configs/pretrain_1.yaml") as file:
        _args = yaml.safe_load(file.read())
        tokenizer_filename = _args["tokenizer_model"]
        training_args = _args["training_args"]
        model_args = _args["model_args"]
        training_settings = _args["training_settings"]
    with open("configs/ps_dataset_1.yaml") as file:
        _args = yaml.safe_load(file.read())
        dataset_path = _args["dataset"]
        dataset_shard_dirpath = _args["shard_directory"]
        tokenizer_setup = _args["tokenizer_args"]
    cached_args = _setup_cached_args(tokenizer_filename, tokenizer_setup, dataset_path)

    # --------- Setup Dataset and Model ---------
    learning_rate = training.CustomSchedule(model_args["model_dim"])
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    pretrainer = _setup_pretrainer(tokenizer_filename, tokenizer_setup, optimizer,
            in_model_dir, cached_args, model_args)

    training_shards, _ = dataset.setup_shards(dataset_shard_dirpath, training_args,
                                                          pretrainer.tokenizer, logger=logger)
    if training_preprocessing is not None:
        training_shards = training_preprocessing(training_shards)

    _batch_shape = (training_args["batch_size"], cached_args["dataset_width"])
    run_test = traintest.build_tester(pretrainer,
            samples=[
                "este é um problema que temos que resolver.",
                "this is a problem we have to solve .",
                "os meus vizinhos ouviram sobre esta ideia.",
                "and my neighboring homes heard about this idea .",
                "vou então muito rapidamente partilhar convosco algumas histórias de algumas "
                "coisas mágicas que aconteceram.",
                "so i 'll just share with you some stories very quickly of some magical things "
                "that have happened .",
                "este é o primeiro livro que eu fiz.",
                "this is the first book i've ever done.",
            ],
            mask_sizes=_batch_shape,
            logger=logger)
    train_batch, train_loss, train_accuracy = traintest.build_pretrainer(
            pretrainer, optimizer, _batch_shape)

    ckpt = tf.train.Checkpoint(optimizer=optimizer, pretrainer=pretrainer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              directory=os.path.join(outdir,
                                                  'checkpoints', ckpt_id),
                                              max_to_keep=50)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logger.info('Latest checkpoint restored!!')

    bucket = training.QuotaBucket(training_settings["skip_bad_loss"]["warmup"],
                                  bucket_capacity=QUOTA_BUCKET_CAPACITY,
                                  bucket_recover=QUOTA_BUCKET_RECOVER,
                                  bucket_recover_rate=QUOTA_BUCKET_RECOVER_RATE)
    prev_losses = training.LossWindow(capacity=SKIP_LOSS_WINDOW)
    nan_reporter = telemetry.detail_reporter(logger, pretrainer.tokenizer)
    if context_rate_overwrite is not None:
        context_rate = context_rate_overwrite
    else:
        context_rate = training_args["context_rate"]
    trainer = traintest.EpochPretrainer(
        traintest.EpochPretrainerInitBuilder().\
            training_settings(training_settings).\
            training_loss(train_loss).\
            training_accuracy(train_accuracy).\
            training_shards(training_shards).\
            training_cb(lambda batch, lengths, loss_check: train_batch(
                batch, lengths, loss_check,
                mask_rate=training_args["mask_rate"],
                context_rate=context_rate)).\
            ckpt_save_cb(lambda _: ckpt_manager.save(options=ckpt_options)).\
            bucket(bucket).\
            prev_losses(prev_losses).build())

    def run_epochs(epochs, skip_shards=None):
        """ run all epoch """
        for epoch in range(epochs):
            start = time.time()
            sublogger = telemetry.PrefixAdapter(logger, 'Epoch {}'.format(epoch+1))
            trainer.run_epoch(skip_shards, logger=sublogger, nan_reporter=nan_reporter)

            if (epoch + 1) % training_settings["epochs_per_save"] == 0:
                logger.info('Saving checkpoint for epoch %d at %s',
                        epoch+1, ckpt_manager.save(options=ckpt_options))

            if (epoch + 1) % training_settings["epochs_per_test"] == 0:
                run_test() # run every epoch

            logger.info('Epoch %d Loss %.4f Accuracy %.4f',
                    epoch+1, train_loss.result(), train_accuracy.result())
            logger.info('Time taken for 1 epoch: %.2f secs', time.time() - start)

    # --------- Training ---------
    run_epochs(nepochs)
    pretrainer.save(os.path.join(outdir, model_id))

if __name__ == '__main__':
    # local logging
    logging.basicConfig(filename="tmp/4_persentence_pretrain_mlm_15p.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    main(_logger, 'export')
