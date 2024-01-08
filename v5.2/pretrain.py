# standard packages
import os
import yaml
import time

# installed packages
import tensorflow as tf
import tensorflow_datasets as tfds

# local packages
import common.training as training
import common.telemetry as telemetry
from common.builder import Builder

import export.model3 as model
import export.training2 as local_training
from export.training import NAN_LOSS_ERR_CODE

from pass_dataset_builder import Builder as PassBuilder

def epoch_pretrainer_init_builder():
    """ create Builder for EpochPretrainer init """
    return Builder(['training_dataset', 'training_loss', 'training'])

class EpochPretrainer:
    """ Reusable pretrainer for running training epochs """
    def __init__(self, args):

        self.training = args['training']
        self.training_dataset = args['training_dataset']
        self.training_loss = args['training_loss']

    def run_epoch(self, logger=None, nan_reporter=None, report_metric=None):
        if logger is None:
            logger = telemetry.EmptyLogger()
        if report_metric is None:
            report_metric = telemetry.get_logger_metric_reporter(logger)

        self.training_loss.reset_states()
        for i, batch in enumerate(self.training_dataset):
            debug_info, err_code = self.training(batch)
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
    def __init__(self, logger, outdir, dataset_config="configs/dataset.yaml"):

        self.logger = logger
        self.outdir = outdir

        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        # --------- Extract configs and cached values ---------
        with open("configs/unet_pretrain.yaml") as file:
            _args = yaml.safe_load(file.read())
            self.model_args = _args["model_args"]

        with open(dataset_config) as file:
            _args = yaml.safe_load(file.read())
            self.builder = PassBuilder() # tfds.builder(_args['tfds_name'])
            self.dataset_args = _args['args']
            self.training_settings = _args['settings']

    def setup_optimizer(self):
        """ create and initialize optimizer """

        learning_rate = training.CustomSchedule(self.model_args["model_dim"])
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate,
                beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        return optimizer

    def setup_unet(self, ckpt_id, ckpt_options=None, in_model_dir=None, nchannels=3):
        """ create and initialize pretrainer """

        optimizer = self.setup_optimizer()
        if ckpt_options is None:
            ckpt_options = tf.train.CheckpointOptions()

        self.logger.info('Model recovered from %s', in_model_dir)
        unet = model.ImageBertPreparer(in_model_dir, nchannels, optimizer, self.model_args)

        ckpt = tf.train.Checkpoint(optimizer=optimizer, unet=unet)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
            directory=os.path.join(self.outdir, 'checkpoints', ckpt_id), max_to_keep=50)

        # if a checkpoint exists, restore the latest checkpoint.
        if in_model_dir is None and ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint, options=ckpt_options)
            self.logger.info('Latest checkpoint restored!!')
        self.logger.info('Beginning training with optimizer iteration=%d', optimizer.iterations.numpy())

        return unet, ckpt_manager, optimizer

    def setup_dataset(self, unet):
        image_size = unet.params['image_dim']
        transform = tf.keras.Sequential([
            tf.keras.layers.Resizing(image_size, image_size),
            tf.keras.layers.CenterCrop(image_size, image_size),
            tf.keras.layers.Activation(lambda x: (x / 127.5) - 1),
        ])

        # dataset
        self.builder.download_and_prepare()
        ds = (self.builder.as_dataset()['train']
                .cache()
                .map(lambda row: transform(row['image']))
                .shuffle(self.dataset_args['buffer_size'])
                .batch(self.dataset_args['batch_size'])
                .prefetch(tf.data.AUTOTUNE))
        return ds

    def e2e(self, in_model_dir,
            ckpt_id='berte_pretrain',
            model_id='berte_pretrain',
            training_preprocessing=None,
            ckpt_options=None,
            report_metric=None):
        """ actually pretrains some model """

        nchannels = 3

        # models
        unet, ckpt_manager, optimizer = self.setup_unet(
                ckpt_id, ckpt_options, in_model_dir, nchannels)

        # dataset
        training_dataset = self.setup_dataset(unet)

        if training_preprocessing is not None:
            training_dataset = training_preprocessing(training_dataset)

        # training
        training_loss = tf.keras.metrics.Mean(name='training_loss')
        unet_train = local_training.UnetTrainer(unet, optimizer, training_loss)

        nan_reporter = telemetry.detail_reporter(self.logger)
        trainer = EpochPretrainer(
            epoch_pretrainer_init_builder().\
                training_loss(training_loss).\
                training_dataset(training_dataset).\
                training(unet_train).build())

        # --------- Training ---------
        nepochs = self.training_settings['epochs']
        for epoch in range(nepochs):
            start = time.time()
            sublogger = telemetry.PrefixAdapter(self.logger, 'Epoch {}'.format(epoch+1))
            trainer.run_epoch(logger=sublogger,
                    nan_reporter=nan_reporter,
                    report_metric=report_metric)

            if (epoch + 1) % self.training_settings["epochs_per_save"] == 0:
                self.logger.info('Saving checkpoint for epoch %d at %s', epoch+1, ckpt_manager.save(options=ckpt_options))


            self.logger.info('Epoch %d Loss %.4f', epoch+1, training_loss.result())
            self.logger.info('Time taken for 1 epoch: %.2f secs', time.time() - start)

        print('saving model')
        unet.save(os.path.join(self.outdir, model_id))
        return unet

