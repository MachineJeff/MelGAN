# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: MelGAN
'''
import tensorflow as tf
from .multiscale import MultiScaleDiscriminator
from .generator import Generator
from .stft_loss import MultiResolutionSTFT
import time
import infolog
import numpy as np
log = infolog.log

class MelGAN():
    def __init__(self, mel_G, audio_G=None):
        self.generator = Generator()
        self.audio_g = audio_G
        self.mel_G = mel_G
        self.G_fake_audio = self.generator(mel_G)

        is_training = audio_G is not None
        if is_training:
            self.discriminator = MultiScaleDiscriminator()

            dis_inputs = tf.concat([self.G_fake_audio, audio_G], axis = 0)
            dis_output = self.discriminator(dis_inputs)

            self.fake_feat = []
            self.real_feat = []

            for feats in dis_output:
                feature_map = feats[0]
                feature_score = feats[1]
                maps = [tf.split(x, num_or_size_splits=2, axis=0) for x in feature_map]
                score = tf.split(feature_score, num_or_size_splits=2, axis=0)
                self.fake_feat.append(([x[0] for x in maps],score[0]))
                self.real_feat.append(([x[1] for x in maps],score[1]))
        
        all_vars = tf.trainable_variables()
        generator_vars = self.generator.vars
        if is_training:
            discriminator_vars = self.discriminator.vars

        log('             MelGAN All Parameters {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars]) / 1000000))
        log('        Generator All Parameters {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in generator_vars]) / 1000000))
        if is_training:
            log('Discriminator All Parameters {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in discriminator_vars]) / 1000000))

    def add_loss(self):
        with tf.variable_scope('gen_loss') as scope:
            self.loss_g = 0.0
            multi_resolution_stft = MultiResolutionSTFT()
            sl_loss, mag_loss = multi_resolution_stft.call(tf.squeeze(self.audio_g, -1), tf.squeeze(self.G_fake_audio, -1))
            stft_loss = 0.5 * (sl_loss + mag_loss)
            for (feats_fake, score_fake), (feats_real, _) in zip(self.fake_feat, self.real_feat):
                self.loss_g += tf.reduce_mean(tf.square(score_fake - 1))
                for feats_f, feats_r in zip(feats_fake, feats_real):
                    self.loss_g += 10 * tf.reduce_mean(tf.abs(feats_f - feats_r))
            self.loss_g += stft_loss

        with tf.variable_scope('dis_loss') as scope:
            self.loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(self.fake_feat, self.real_feat):
                self.loss_d += tf.reduce_mean(tf.square(score_real - 1))
                self.loss_d += tf.reduce_mean(tf.square(score_fake - 0))

    def add_optimizer(self, global_step):
        self.learning_rate = self._learning_rate_decay(global_step)

        with tf.variable_scope('gen_opt') as scope:
            opt_g = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9)
            gradients_g, variables_g = zip(*opt_g.compute_gradients(self.loss_g, var_list = self.generator.vars))
            clipped_gradients_g, _ = tf.clip_by_global_norm(gradients_g, 10.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize_g = opt_g.apply_gradients(zip(clipped_gradients_g, variables_g), global_step=global_step)

            # self.optimize_g = tf.train.AdamOptimizer(0.0001, 0.5, 0.9).minimize(self.loss_g, var_list = self.generator.vars, global_step = global_step)

        with tf.variable_scope('dis_opt') as scope:
            opt_d = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9)
            gradients_d, variables_d = zip(*opt_d.compute_gradients(self.loss_d, var_list = self.discriminator.vars))
            clipped_gradients_d, _ = tf.clip_by_global_norm(gradients_d, 10.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize_d = opt_d.apply_gradients(zip(clipped_gradients_d, variables_d), global_step=global_step)

            # self.optimize_d = tf.train.AdamOptimizer(0.0001, 0.5, 0.9).minimize(self.loss_d, var_list = self.discriminator.vars)
            
    def _learning_rate_decay(self, global_step):
        #Compute natural exponential decay
        lr = tf.train.exponential_decay(
            1e-4, 
            global_step - 600000, #lr = 1e-3 at step 50k
            300000, 
            0.63, #lr = 1e-5 around step 310k
            name='lr_exponential_decay')

                #clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, 1e-6), 1e-4)
