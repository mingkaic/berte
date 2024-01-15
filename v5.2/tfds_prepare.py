# standard packages
import math
import yaml

# installed packages
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def main(dataset_config="configs/dataset.yaml", nshards=24):

    # prepare builder
    with open(dataset_config) as file:
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

    # number of shards
    shard_size = math.ceil(n / nshards)
    for i in range(n):
        print('saving shard {}'.format(i))
        tf.data.experimental.save(ds.skip(i * shard_size), 'intake/tfds_{}_{}'.format(ds_name, i))
        print('done saving shard {}'.format(i))

if __name__ == '__main__':
    main()
