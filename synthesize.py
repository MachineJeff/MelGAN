# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: 
'''
import tensorflow as tf
from infolog import log
import numpy as np
from models.melgan import MelGAN
import time
class Synthesizer:
    def load(self, checkpoint_path):
        log('Constructing model: MelGAN')
        input_mel = tf.placeholder(tf.float32, [1, None, 80], name = 'mel_G')
        with tf.variable_scope('MelGAN') as scope:
            self.model = MelGAN(input_mel)
            self.audio_output = self.model.G_fake_audio
        log('Loading checkpoint from {}'.format(checkpoint_path))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config = config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, mel_file, path = None):
        mel = np.load(mel_file)
        mel = np.expand_dims(mel, axis=0)

        feed_dict = {self.model.mel_G: mel}
        t1 = time.time()
        out = self.session.run(self.audio_output, feed_dict = feed_dict)
        t2 = time.time()
        print('cost:{}'.format(t2-t1))
        out = np.squeeze(out, 0)
        out = np.squeeze(out, 1)
        return out
