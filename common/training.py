#!/usr/bin/env python3
"""
This module includes functions that trains models
"""

import numpy as np

import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Customer schedule for BERT learning rate
    """
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
                "d_model": self.d_model,
                "warmup_steps": self.warmup_steps,
                }

class QuotaBucket:
    """
    QuotaBucket tracks failure.
    """
    def __init__(self, warmup, bucket_capacity, bucket_recover, bucket_recover_rate):

        self.warmup = warmup
        self.quota = bucket_capacity
        self.iter = 0

        self.bucket_capacity = bucket_capacity
        self.bucket_recover = bucket_recover
        self.bucket_recover_rate = bucket_recover_rate

    def can_skip(self):
        """
        Return True if bucket still in warmup phase
        """
        if self.iter < self.warmup:
            return True
        return False

    def have_quota(self):
        """
        Return True if bucket still have quota
        """
        return self.quota > 0

    def update(self, failed):
        """
        Increment counters depending on whether failed.
        If failed, quota depletes but never below 0.
        """
        self.iter += 1
        if self.iter % self.bucket_recover_rate == 0:
            self.quota = max(self.quota + self.bucket_recover, self.bucket_capacity)

        if failed and self.quota > 0:
            self.quota -= 1


class LossWindow:
    """
    LossWindow is a sliding window.

    Args:
    capacity: the number of values to remember in the window.
    """
    def __init__(self, capacity):

        self.window = []
        self.capacity = capacity

    def __call__(self,
                 default_value,
                 stdev_coeff=1.):
        """
        Return a value statistically determined by the window values and the args.

        Args:
        default_value: the "loss benchmark" when window is empty.
        stdev_coeff: the "loss benchmark" is mean + stdev_coeff * stdev.
                     E.g. when stdev_coeff=1, return the worst ~15%-tile of the window
        """
        if len(self.window) == 0:
            return default_value

        mean = np.mean(self.window)
        stdev = np.std(self.window)
        return tf.constant(mean + stdev_coeff * stdev, dtype=tf.float32)

    def add(self, loss):
        """
        Add loss value to the window.
        """

        self.window.append(loss)
        if len(self.window) > self.capacity:
            self.window.pop(0)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred, pad=0):
    """
    common loss function used when training
    """

    mask = tf.math.logical_not(tf.math.equal(real, tf.cast(pad, real.dtype)))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred, pred_axis=2):
    """
    accuracy function used during training
    """

    return equivalent_accuracy_function(real, tf.argmax(pred, axis=pred_axis))

def equivalent_accuracy_function(real, pred):
    """
    accuracy function between two tensors of equal shape used during training
    """

    accuracies = tf.equal(real, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
