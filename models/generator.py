# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: Generator
'''

import tensorflow as tf
from .convolutions import wn_conv1d, wn_deconv1d
import infolog

log = infolog.log

class ResStack:
    def __init__(self, channel):
        super(ResStack, self).__init__()
        self.channel = channel

    def __call__(self, inputs, scope):
        # wn_conv1d(x, kernel_size, channels, scope, stride=1, pad='same', dilation=1, groups=1)
        out = inputs
        for i in range(3):
            temp = out
            out = tf.nn.leaky_relu(out, 0.02)
            out = wn_conv1d(out, 3, self.channel, str(scope) + '_{}_dia_conv'.format(i), dilation = 3**i)

            out = tf.nn.leaky_relu(out, 0.02)            
            out = wn_conv1d(out, 3, self.channel, str(scope) + '_{}_conv'.format(i))

            out += temp
        return out


class Generator():
    def __init__(self):
        self.name = 'Generator'
        self.res_stack1 = ResStack(256)
        self.res_stack2 = ResStack(128)
        self.res_stack3 = ResStack(64)
        self.res_stack4 = ResStack(32)
    def __call__(self, mel):
        # wn_conv1d(x, kernel_size, channels, scope, stride=1, pad='same', dilation=1, groups=1)
        # wn_deconv1d(x, kernel_size, channels, scope, stride=1, pad='same')
        with tf.variable_scope(self.name) as scope:
            out = wn_conv1d(mel, 7, 512, self.name + '_1thconv1d', 1)
    
            out = tf.nn.leaky_relu(out, 0.02)
            out = wn_deconv1d(out, 10, 256, self.name + '_1thdeconv1d', 5)
    
            out = self.res_stack1(out, scope = self.name + '_1thresstack')
    
            out = tf.nn.leaky_relu(out, 0.02)
            out = wn_deconv1d(out, 10, 128, self.name + '_2thdeconv1d', 5)
            
            out = self.res_stack2(out, scope = self.name + '_2thresstack')
            
            out = tf.nn.leaky_relu(out, 0.02)
            out = wn_deconv1d(out, 8, 64, self.name + '_3thdeconv1d', 4)    
            
            out = self.res_stack3(out, scope = self.name + '_3thresstack')
            
            out = tf.nn.leaky_relu(out, 0.02)
            out = wn_deconv1d(out, 4, 32, self.name + '_4thdeconv1d', 2)    
            
            out = self.res_stack4(out, scope = self.name + '_4thresstack')
            
            out = tf.nn.leaky_relu(out, 0.02)
            out = wn_conv1d(out, 7, 1, self.name + '_last_conv1d')
            
            audio = tf.nn.tanh(out)

            return audio
                
    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]
