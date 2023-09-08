#!/usr/bin/env python3
"""
This module defines pretrain_mlm.ipynb logic with dataset
"""
# standard packages
import time
import logging
import os

import functools
import yaml

# installed packages
import tensorflow_text as text
import tensorflow as tf

# local packages
import export.dataset as dataset
import export.mlm as mlm
import export.model5 as model
from export.model import InitParamBuilder

import common.telemetry as telemetry
import common.training as training
from common.cache import cache_values

EPOCH_PRETRAINER_ARGS = [
    "training_settings",
    "training_loss",
    "training_accuracy",
    "training_batches",
    "training_cb",
    "ckpt_save_cb",
]

def _setup_cached_args(tokenizer_filename, tokenizer_setup, dataset_path):
    with open(tokenizer_filename, 'rb') as file:
        tokenizer = text.SentencepieceTokenizer(model=file.read(),
                                                out_type=tf.int32,
                                                add_bos=tokenizer_setup["add_bos"],
                                                add_eos=tokenizer_setup["add_eos"])
    # generate or retrieve cached values
    return cache_values("configs/pretrain_mlm_cache.yaml", {
        "dataset_width": dataset.generate_dataset_width,
        "vocab_size": lambda _, tokenizer: int(tokenizer.vocab_size().numpy()),
    }, dataset_path, tokenizer)

class EpochPretrainerInitBuilder:
    """ build pretrainer init arguments """
    def __init__(self):
        self.arg = dict()
        for key in EPOCH_PRETRAINER_ARGS:
            setattr(self.__class__, key, functools.partial(self.add_arg, key=key))

    def add_arg(self, value, key):
        """ add custom arg """
        self.arg[key] = value
        return self

    def build(self):
        """ return built arg """
        return self.arg

class EpochPretrainer:
    """ Reusable pretrainer for running training epochs """
    def __init__(self, args):

        self.args = args
        self.training_batches = args["training_batches"]

        self.training_loss = args["training_loss"]
        self.training_accuracy = args.get("training_accuracy", None)

    def run_epoch(self, logger=None, nan_reporter=None, report_metric=None):
        """ run a single epoch """
        if logger is None:
            logger = telemetry.EmptyLogger()
        if report_metric is None:
            report_metric = telemetry.get_logger_metric_reporter(logger)

        self.training_loss.reset_states()
        if self.training_accuracy is not None:
            self.training_accuracy.reset_states()
        for i, (batch, lengths) in enumerate(self.training_batches):
            debug_info, err_code = self.args["training_cb"](batch, lengths)
            if err_code == mlm.NAN_LOSS_ERR_CODE:
                if nan_reporter is not None:
                    nan_reporter(debug_info)
                logger.error('batch %d produced nan loss! skipping...', i)
                # nan is fatal
                return

            if i % 50 == 0:
                report_metric(
                        Batch=i,
                        Loss=float(self.training_loss.result().numpy()),
                        Accuracy=float(self.training_accuracy.result().numpy()))

class PretrainerPipeline:
    """ pretrainer pipeline """
    def __init__(self, logger, outdir):

        self.logger = logger
        self.outdir = outdir

        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        # --------- Extract configs and cached values ---------
        with open("configs/pretrain.yaml") as file:
            _args = yaml.safe_load(file.read())
            self.tokenizer_setup = _args["tokenizer_args"]
            self.model_args = _args["model_args"]
            self.preprocessor_dir = _args["preprocessor"]
        with open("configs/mlm_dataset.yaml") as file:
            _args = yaml.safe_load(file.read())
            self.tokenizer_filename = _args["tokenizer_model"]
            self.dataset_path = _args["path"]
            self.training_args = _args["args"]
            self.training_settings = _args["settings"]
        self.cached_args = _setup_cached_args(
            self.tokenizer_filename, self.tokenizer_setup, self.dataset_path)

    def setup_optimizer(self):
        """ create and initialize optimizer """

        learning_rate = training.CustomSchedule(self.model_args["model_dim"])
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate,
                beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        return optimizer

    def setup_pretrainer(self, ckpt_id,
            in_model_dir = None,
            ckpt_options=None):
        """ create and initialize pretrainer """

        optimizer = self.setup_optimizer()
        if ckpt_options is None:
            ckpt_options = tf.train.CheckpointOptions()

        self.logger.info('Model recovered from %s', in_model_dir)
        builder = InitParamBuilder()
        _args = dict(self.cached_args)
        _args.update(self.model_args)
        for key in _args:
            builder.add_param(_args[key], key)
        preprocessor = model.PretrainerPreprocessor(self.preprocessor_dir,
                optimizer, builder.build(), self.tokenizer_setup)
        pretrainer = model.ExtendedPretrainerMLM(in_model_dir, builder.build())

        ckpt = tf.train.Checkpoint(
                optimizer=optimizer,
                preprocessor=preprocessor,
                pretrainer=pretrainer)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
            directory=os.path.join(self.outdir, 'checkpoints', ckpt_id), max_to_keep=50)

        # if a checkpoint exists, restore the latest checkpoint.
        if in_model_dir is None and ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint, options=ckpt_options)
            self.logger.info('Latest checkpoint restored!!')
        self.logger.info('Beginning training with optimizer iteration=%d',
            optimizer.iterations.numpy())

        return preprocessor, pretrainer, ckpt_manager, optimizer

    def e2e(self,
            in_model_dir=None,
            ckpt_id='berte_pretrain',
            model_id='berte_pretrain',
            training_preprocessing=None,
            ckpt_options=None,
            report_metric=None):
        """ actually pretrains some model """

        # models
        preprocessor, pretrainer, ckpt_manager, optimizer = self.setup_pretrainer(
                ckpt_id, in_model_dir, ckpt_options)

        # testing
        run_test = mlm.build_tester(preprocessor, pretrainer,
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
        training_batches = dataset.setup_mlm_dataset(self.dataset_path, self.training_args,
                preprocessor.tokenizer, logger=self.logger)
        if training_preprocessing is not None:
            training_batches = training_preprocessing(training_batches)

        # training
        train_batch, train_loss, train_accuracy = mlm.build_pretrainer(preprocessor,
                pretrainer, optimizer, self.cached_args["dataset_width"])

        nan_reporter = telemetry.detail_reporter(self.logger)
        trainer = EpochPretrainer(
            EpochPretrainerInitBuilder().\
                training_settings(self.training_settings).\
                training_loss(train_loss).\
                training_accuracy(train_accuracy).\
                training_batches(training_batches).\
                training_cb(lambda batch, lengths: train_batch(batch, lengths,
                    mask_rate=self.training_args["mask_rate"])).\
                ckpt_save_cb(lambda _: ckpt_manager.save(options=ckpt_options)).build())

        # --------- Training ---------
        for epoch in range(self.training_settings["epochs"]):
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
    logging.basicConfig(filename="tmp/pretrain.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    PretrainerPipeline(_logger, 'export').e2e(in_model_dir='intake/berte_pretrain')
