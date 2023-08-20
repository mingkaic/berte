"""
This module defines pretrain_nsp.ipynb logic with dataset
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
import dataset
import pretrain_epoch

import export.nsp as nsp
import export.model3 as model

import common.telemetry as telemetry
import common.training as training
from common.cache import cache_values

def _setup_cached_args(tokenizer_filename, tokenizer_setup, dataset_path):
    with open(tokenizer_filename, 'rb') as file:
        tokenizer = text.SentencepieceTokenizer(model=file.read(),
                                                out_type=tf.int32,
                                                add_bos=tokenizer_setup["add_bos"],
                                                add_eos=tokenizer_setup["add_eos"])
    # generate or retrieve cached values
    return cache_values("configs/pretrain_cache.yaml", {
        "vocab_size": lambda _, tokenizer: int(tokenizer.vocab_size().numpy()),
    }, dataset_path, tokenizer)

class PretrainerPipeline:
    """ pretrainer pipeline """
    def __init__(self, logger, outdir):

        self.logger = logger
        self.outdir = outdir

        # --------- Extract configs and cached values ---------
        with open("configs/pretrain.yaml") as file:
            _args = yaml.safe_load(file.read())
            self.tokenizer_filename = _args["tokenizer_model"]
            self.training_args = _args["training_args"]
            self.model_args = _args["model_args"]
            self.training_settings = _args["training_settings"]
        with open("configs/ps_dataset.yaml") as file:
            _args = yaml.safe_load(file.read())
            self.dataset_path = _args["dataset"]
            self.tokenizer_setup = _args["tokenizer_args"]
        self.cached_args = _setup_cached_args(
            self.tokenizer_filename, self.tokenizer_setup, self.dataset_path)

    def setup_optimizer(self):
        """ create and initialize optimizer """

        learning_rate = training.CustomSchedule(self.model_args["model_dim"])
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate,
                beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        return optimizer

    def setup_pretrainer(self, optimizer, ckpt_id, in_model_dir, ckpt_options=None):
        """ create and initialize pretrainer """

        if ckpt_options is None:
            ckpt_options = tf.train.CheckpointOptions()

        self.logger.info('Model recovered from %s', in_model_dir)
        builder = model.InitParamBuilder()
        _args = dict(self.cached_args)
        _args.update(self.model_args)
        for key in _args:
            builder.add_param(_args[key], key)
        pretrainer = model.PretrainerNSP(in_model_dir,
                optimizer, builder.build(), self.tokenizer_setup)

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
            ckpt_id='berte_pretrain',
            model_id='berte_pretrain',
            training_preprocessing=None,
            ckpt_options=None,
            report_metric=None):
        """ actually pretrains some model """

        # models
        optimizer = self.setup_optimizer()
        pretrainer, ckpt_manager = self.setup_pretrainer(optimizer, ckpt_id, in_model_dir, ckpt_options)
        self.logger.info('Beginning training with optimizer iteration=%d',
            optimizer.iterations.numpy())

        # testing
        run_test = nsp.build_tester(pretrainer, paragraph=[
            "the history of raja ampat is steeped in history and mythology.",
            "raja ampat was originally a part of an influential southeastern sultanate, the sultanate of tidore.",
            "in 1562, it was captured by the dutch, who influenced much of its cuisines and its architecture.",
            "that’s the more boring origin story.",
            "a colorful local tale talks of a woman who was given seven precious eggs, of which four (‘ampat’ in bahasa indonesian) hatched into kings (‘raja’).",
        ], logger=self.logger)

        training_batches = dataset.setup_dataset(
                self.dataset_path, self.training_args, logger=self.logger)
        if training_preprocessing is not None:
            training_batches = training_preprocessing(training_batches)

        # training
        train_batch, train_loss, train_accuracy = nsp.build_pretrainer(
                pretrainer, optimizer, self.training_args['batch_size'])

        nan_reporter = telemetry.detail_reporter(self.logger)
        trainer = pretrain_epoch.EpochPretrainer(
            pretrain_epoch.EpochPretrainerInitBuilder().\
                training_settings(self.training_settings).\
                training_loss(train_loss).\
                training_accuracy(train_accuracy).\
                training_batches(training_batches).\
                training_cb(train_batch).\
                ckpt_save_cb(lambda _: ckpt_manager.save(options=ckpt_options)).build())

        # --------- Training ---------
        for epoch in range(nepochs):
            start = time.time()
            sublogger = telemetry.PrefixAdapter(self.logger, 'Epoch {}'.format(epoch+1))
            trainer.run_epoch(logger=sublogger, nan_reporter=nan_reporter,
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
    logging.basicConfig(filename="tmp/pretrain_nsp.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    PretrainerPipeline(_logger, 'export').e2e()
