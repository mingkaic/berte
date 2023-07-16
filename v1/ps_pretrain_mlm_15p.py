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

class PretrainerPipeline:
    """ pretrainer pipeline """
    def __init__(self, logger, outdir):

        self.logger = logger
        self.outdir = outdir

        # --------- Extract configs and cached values ---------
        with open("configs/pretrain_1.yaml") as file:
            _args = yaml.safe_load(file.read())
            self.tokenizer_filename = _args["tokenizer_model"]
            self.training_args = _args["training_args"]
            self.model_args = _args["model_args"]
            self.training_settings = _args["training_settings"]
        with open("configs/ps_dataset_1.yaml") as file:
            _args = yaml.safe_load(file.read())
            self.dataset_path = _args["dataset"]
            self.dataset_shard_dirpath = _args["shard_directory"]
            self.tokenizer_setup = _args["tokenizer_args"]
        self.cached_args = _setup_cached_args(
            self.tokenizer_filename, self.tokenizer_setup, self.dataset_path)

    def setup_optimizer(self, optimizer_it):
        """ create and initialize optimizer """

        learning_rate = training.CustomSchedule(self.model_args["model_dim"])
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate,
                beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        optimizer.iterations.assign(optimizer_it)
        return optimizer

    def setup_pretrainer(self, optimizer, ckpt_id,
                        ckpt_options=None,
                        in_model_dir=None):
        """ create and initialize pretrainer """

        if ckpt_options is None:
            ckpt_options = tf.train.CheckpointOptions()

        if in_model_dir is not None:
            self.logger.info('Model recovered from %s', in_model_dir)
            pretrainer = models.PretrainerMLM.load(in_model_dir, optimizer,
                    self.tokenizer_filename, self.tokenizer_setup)
        else:
            with open(self.tokenizer_filename, 'rb') as file:
                tokenizer = text.SentencepieceTokenizer(model=file.read(),
                                                        out_type=tf.int32,
                                                        add_bos=self.tokenizer_setup["add_bos"],
                                                        add_eos=self.tokenizer_setup["add_eos"])
            metadata = models.BerteMetadata(self.tokenizer_filename, optimizer.iterations)
            builder = models.InitParamBuilder()
            _args = dict(self.cached_args)
            _args.update(self.model_args)
            for key in _args:
                builder.add_param(_args[key], key)
            pretrainer = models.PretrainerMLM(
                tokenizer=tokenizer,
                params=builder.build(),
                metadata=metadata)

        ckpt = tf.train.Checkpoint(optimizer=optimizer, pretrainer=pretrainer)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
            directory=os.path.join(self.outdir, 'checkpoints', ckpt_id), max_to_keep=50)

        # if a checkpoint exists, restore the latest checkpoint.
        if in_model_dir is None and ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint, options=ckpt_options)
            self.logger.info('Latest checkpoint restored!!')

        return pretrainer, ckpt_manager

    def e2e(self,
            nepochs=1,
            in_model_dir=None,
            ckpt_id='train_15p_ps',
            model_id='berte_pretrain_mlm_15p',
            training_preprocessing=None,
            context_rate_overwrite=None,
            ckpt_options=None,
            skip_shards=None,
            skip_loss=True,
            optimizer_it=0,
            report_metric=None):
        """ actually pretrains some model """

        # models
        optimizer = self.setup_optimizer(optimizer_it)
        pretrainer, ckpt_manager = self.setup_pretrainer(optimizer, ckpt_id,
                ckpt_options, in_model_dir)

        # testing
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
                mask_sizes=(self.training_args["batch_size"], self.cached_args["dataset_width"]),
                logger=self.logger)

        # dataset
        training_shards, _ = dataset.setup_shards(self.dataset_shard_dirpath, self.training_args,
                pretrainer.tokenizer, logger=self.logger)
        if training_preprocessing is not None:
            training_shards = training_preprocessing(training_shards)

        # training
        train_batch, train_loss, train_accuracy = traintest.build_pretrainer(
                pretrainer, optimizer, self.cached_args["dataset_width"], skip_loss)

        bucket = training.QuotaBucket(self.training_settings["skip_bad_loss"]["warmup"],
                                      bucket_capacity=QUOTA_BUCKET_CAPACITY,
                                      bucket_recover=QUOTA_BUCKET_RECOVER,
                                      bucket_recover_rate=QUOTA_BUCKET_RECOVER_RATE)
        prev_losses = training.LossWindow(capacity=SKIP_LOSS_WINDOW)
        nan_reporter = telemetry.detail_reporter(self.logger, pretrainer.tokenizer)
        if context_rate_overwrite is not None:
            context_rate = context_rate_overwrite
        else:
            context_rate = self.training_args["context_rate"]
        trainer = traintest.EpochPretrainer(
            traintest.EpochPretrainerInitBuilder().\
                training_settings(self.training_settings).\
                training_loss(train_loss).\
                training_accuracy(train_accuracy).\
                training_shards(training_shards).\
                training_cb(lambda batch, lengths, loss_check: train_batch(
                    batch, lengths, loss_check,
                    mask_rate=self.training_args["mask_rate"],
                    context_rate=context_rate)).\
                ckpt_save_cb(lambda _: ckpt_manager.save(options=ckpt_options)).\
                bucket(bucket).\
                prev_losses(prev_losses).build())

        # --------- Training ---------
        for epoch in range(nepochs):
            start = time.time()
            sublogger = telemetry.PrefixAdapter(self.logger, 'Epoch {}'.format(epoch+1))
            trainer.run_epoch(skip_shards, logger=sublogger, nan_reporter=nan_reporter,
                report_metric=report_metric)

            if (epoch + 1) % self.training_settings["epochs_per_save"] == 0:
                self.logger.info('Saving checkpoint for epoch %d at %s',
                        epoch+1, ckpt_manager.save(options=ckpt_options))

            if (epoch + 1) % self.training_settings["epochs_per_test"] == 0:
                run_test() # run every epoch

            self.logger.info('Epoch %d Loss %.4f Accuracy %.4f',
                    epoch+1, train_loss.result(), train_accuracy.result())
            self.logger.info('Time taken for 1 epoch: %.2f secs', time.time() - start)

        pretrainer.save(os.path.join(self.outdir, model_id))
        return pretrainer

if __name__ == '__main__':
    # local logging
    logging.basicConfig(filename="tmp/4_persentence_pretrain_mlm_15p.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    PretrainerPipeline(_logger, 'export').e2e()
