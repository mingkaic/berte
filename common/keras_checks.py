"""
This module checks keras objects
"""
import tensorflow as tf

def assert_model_eq(lobj, robj):
    """ assert keras models equality """

    assert isinstance(lobj, tf.keras.Model) and isinstance(robj, tf.keras.Model)
    lweights = {weight.name:weight for weight in lobj.weights}
    rweights = {weight.name:weight for weight in robj.weights}
    assert len(lweights) == len(rweights)
    for weightname in lweights:
        assert weightname in rweights
        lweight = lweights[weightname].numpy()
        rweight = rweights[weightname].numpy()
        assert (lweight == rweight).all()
