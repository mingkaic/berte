"""
This module pretrain memory processor model using the nsp method
"""
# standard packages
import time
import logging
import os.path

import functools
import yaml

# installed packages
import tensorflow as tf

# local packages
import export.dataset as dataset
import export.nsp as nsp
import export.model as model
from export.mlm import NAN_LOSS_ERR_CODE

import common.telemetry as telemetry
import common.training as training

EPOCH_PRETRAINER_ARGS = [
    "training_settings",
    "training_loss",
    "training_accuracy",
    "training_batches",
    "training_cb",
    "ckpt_save_cb",
]

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
        for i, batch in enumerate(self.training_batches):
            debug_info, err_code = self.args["training_cb"](batch)
            if err_code == NAN_LOSS_ERR_CODE:
                if nan_reporter is not None:
                    nan_reporter(debug_info)
                logger.error('batch %d produced nan loss! skipping...', i)
                # nan is fatal
                return

            if i % 50 == 0:
                report_metric(
                    Batch=i,
                    Loss=float(self.training_loss.result().numpy()))

class PretrainerPipeline:
    """ pretrainer pipeline """
    def __init__(self, logger, outdir,
            dataset_config="configs/nsp_dataset.yaml"):

        self.logger = logger
        self.outdir = outdir

        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        # --------- Extract configs and cached values ---------
        with open("configs/pretrain.yaml") as file:
            _args = yaml.safe_load(file.read())
            self.tokenizer_setup = _args["tokenizer_args"]
            self.model_args = _args["model_args"]
        with open(dataset_config) as file:
            _args = yaml.safe_load(file.read())
            self.dataset_path = _args["path"]
            self.training_args = _args["args"]
            self.training_settings = _args["settings"]

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
        for key in self.model_args:
            builder.add_param(self.model_args[key], key)
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

        training_batches = dataset.setup_nsp_dataset(
                self.dataset_path, self.training_args, logger=self.logger)
        if training_preprocessing is not None:
            training_batches = training_preprocessing(training_batches)

        # training
        train_batch, train_loss = nsp.build_pretrainer(pretrainer, optimizer)

        nan_reporter = telemetry.detail_reporter(self.logger)
        trainer = EpochPretrainer(
            EpochPretrainerInitBuilder().\
                training_settings(self.training_settings).\
                training_loss(train_loss).\
                training_batches(training_batches).\
                training_cb(train_batch).\
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

            self.logger.info('Epoch %d Loss %.4f',
                    epoch+1, train_loss.result())
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
