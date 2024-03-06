# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: Discriminator
'''
from .convolutions import wn_conv1d
import tensorflow as tf
import infolog

log = infolog.log

class Discriminator():
    def __init__(self, index):
        self.index = index
    def __call__(self, inputs, scope):
        # wn_conv1d(x, kernel_size, channels, scope, stride=1, pad='same', dilation=1, groups=1)
        feature_map = list()

        out = wn_conv1d(inputs, 15, 16, str(scope) + '_1th_conv1d')
        out = tf.nn.leaky_relu(out, 0.02)
        feature_map.append(out)

        out = wn_conv1d(out, 41, 64, str(scope) + '_1th_sepconv1d', stride = 4, groups = 4)
        out = tf.nn.leaky_relu(out, 0.02)
        feature_map.append(out)

        out = wn_conv1d(out, 41, 256, str(scope) + '_2th_sepconv1d', stride = 4, groups = 16)
        out = tf.nn.leaky_relu(out, 0.02)
        feature_map.append(out)

        out = wn_conv1d(out, 41, 1024, str(scope) + '_3th_sepconv1d', stride = 4, groups = 64)
        out = tf.nn.leaky_relu(out, 0.02)
        feature_map.append(out)

        out = wn_conv1d(out, 41, 1024, str(scope) + '_4th_sepconv1d', stride = 4, groups = 256)
        out = tf.nn.leaky_relu(out, 0.02)
        feature_map.append(out)

        out = wn_conv1d(out, 5, 1024, str(scope) + '_2th_conv1d')
        out = tf.nn.leaky_relu(out, 0.02)
        feature_map.append(out)

        score = wn_conv1d(out, 3, 1, str(scope) + '_last_conv1d')

        return feature_map, score
