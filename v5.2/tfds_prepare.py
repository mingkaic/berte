# standard packages
import argparse
import os
import math
import yaml

# installed packages
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def main():
    """ main """
    parser = argparse.ArgumentParser(
                    prog='tfds_prepare',
                    description='split dataset specified by configs into different shards')
    parser.add_argument('--config', dest='config', default='configs/dataset.yaml')
    parser.add_argument('--nshards', dest='nshards', default=24)
    parser.add_argument('--shard_index', dest='shard_index', default=-1)
    parser.add_argument('--output_dir', dest='output_dir', default='intake1')

    args = parser.parse_args()

    # prepare builder
    with open(args.dataset_config) as file:
        _args = yaml.safe_load(file.read())
        ds_name = _args['tfds_name']
        builder = tfds.builder(ds_name)
    builder.download_and_prepare()

    # transform image in order to save on memory size
    with open("configs/unet_pretrain.yaml") as file:
        _args = yaml.safe_load(file.read())
        image_size = _args["model_args"]["image_dim"]
    transform = tf.keras.Sequential([
        tf.keras.layers.Resizing(image_size, image_size),
        tf.keras.layers.CenterCrop(image_size, image_size),
        tf.keras.layers.Activation(lambda x: (x / 127.5) - 1),
    ])

    # dataset
    ds = (builder.as_dataset()['train']
            .cache()
            .map(lambda row: transform(row['image']))
            .prefetch(tf.data.AUTOTUNE))
    n = ds.cardinality().numpy()

    # save shards
    shard_size = math.ceil(n / args.nshards)
    if args.shard_index < 0:
        indices = range(args.nshards)
    else:
        indices = [args.shard_index]
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for i in indices:
        print('saving shard {}'.format(i))
        ds.skip(i * shard_size).\
                take(shard_size).\
                save(os.path.join(args.output_dir, 'tfds_{}_{}'.format(ds_name, i)))
        print('done saving shard {}'.format(i))

if __name__ == '__main__':

    main()
