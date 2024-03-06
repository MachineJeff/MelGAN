# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: MultiScaleDiscriminator
'''
from .discriminator import Discriminator
import tensorflow as tf


class Pool():
    def __init__(self):
        pass
    def __call__(self, inputs):
        out = tf.layers.average_pooling1d(
          inputs = inputs,
          pool_size = 4,
          strides = 2,
          padding='same')
        return out


class Identity():
    def __init__(self):
        pass
    def __call__(self, inputs):
        return inputs


class MultiScaleDiscriminator():
    def __init__(self):
        self.discriminators = [Discriminator(i) for i in range(3)]
        self.poolings = [Identity()] + [Pool() for i in range(2)]
        self.name = 'MultiScaleDiscriminator'

    def __call__(self, inputs):
        with tf.variable_scope(self.name) as scope:
            ret = list()
            head = 0
            for pool, disc in zip(self.poolings, self.discriminators):
                inputs = pool(inputs)
                ret.append(disc(inputs, scope='disc_{}'.format(head)))
                head += 1

            return ret # [(feat, score), (feat, score), (feat, score)]

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]