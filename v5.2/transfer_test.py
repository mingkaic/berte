import os
import logging

import tensorflow as tf

import export.model3 as model

import common.training as training

import transfer

if __name__ == '__main__':
    MODEL = 'intake/berte_pretrain'
    TMP_DIR = 'transfer_tmp'

    os.makedirs(TMP_DIR, exist_ok=True)
    logging.basicConfig(filename=os.path.join(TMP_DIR, 'converter.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    learning_rate = training.CustomSchedule(256)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate,
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    logdir = os.path.join(TMP_DIR, 'textbert')
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)

    bert = model.TextBert(MODEL, optimizer, None, None)

    # forward pass
    run_test = transfer.build_tester(bert, paragraph=[
        "the history of raja ampat is steeped in history and mythology.",
        "raja ampat was originally a part of an influential southeastern sultanate, the sultanate of tidore.",
        "in 1562, it was captured by the dutch, who influenced much of its cuisines and its architecture.",
        "that’s the more boring origin story.",
        "a colorful local tale talks of a woman who was given seven precious eggs, of which four (‘ampat’ in bahasa indonesian) hatched into kings (‘raja’).",
    ], logger=logger)
    run_test()

    with writer.as_default():
        tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=logdir)
