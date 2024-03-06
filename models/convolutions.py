# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: Custom Convolutions
'''
import tensorflow as tf
from .normalization import WeightNorm
from models.group_conv import *

def wn_conv1d(x, kernel_size, channels, scope, stride=1, pad='same', dilation=1, groups=1):
    with tf.variable_scope(scope):
        output = WeightNorm(group_Conv1D(
          filters = channels, 
          kernel_size = kernel_size, 
          strides = stride, 
          padding = pad, 
          dilation_rate = dilation,
          groups = groups))(x)
        return output


def wn_deconv1d(x, kernel_size, channels, scope, stride=1, pad='same'):
    x = tf.expand_dims(x, 1)
    kernel_size = (1, kernel_size)
    stride = (1, stride)
    with tf.variable_scope(scope):
        output = WeightNorm(tf.layers.Conv2DTranspose(
          filters = channels, 
          kernel_size = kernel_size, 
          strides = stride, 
          padding = pad))(x)
        output = tf.squeeze(output, 1)
        return output
