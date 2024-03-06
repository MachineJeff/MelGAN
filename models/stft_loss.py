# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: loss
'''
import tensorflow as tf


class STFT:
    def __init__(self, frame_length=600, frame_step=120, fft_length=1024, scope=None):
        super(STFT, self).__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.scope = 'stft' if scope is None else scope
        
    def spectral_convergene_loss(self, y_mag, x_mag):
        return tf.norm(y_mag - x_mag, ord='fro', axis=(-2, -1)) / tf.norm(y_mag, ord="fro", axis=(-2, -1))

    def log_stft_magnitude_loss(self, y_mag, x_mag):
        return tf.abs(tf.math.log(y_mag) - tf.math.log(x_mag))

    def call(self, y, x):
        with tf.variable_scope(self.scope):
            x_mag = tf.abs(tf.contrib.signal.stft(signals=x,
                                                  frame_length=self.frame_length,
                                                  frame_step=self.frame_step,
                                                  fft_length=self.fft_length))
            y_mag = tf.abs(tf.contrib.signal.stft(signals=y,
                                                  frame_length=self.frame_length,
                                                  frame_step=self.frame_step,
                                                  fft_length=self.fft_length))
            x_mag = tf.sqrt(x_mag ** 2 + 1e-7)
            y_mag = tf.sqrt(y_mag ** 2 + 1e-7)

            sc_loss = self.spectral_convergene_loss(y_mag, x_mag)
            mag_loss = self.log_stft_magnitude_loss(y_mag, x_mag)

            return sc_loss, mag_loss


class MultiResolutionSTFT:
    '''Multi resolution STFT loss module'''

    def __init__(self, fft_lengths=[1024, 2048, 512], frame_lengths=[600, 1200, 240], frame_steps=[120, 240, 50],
                 scope=None):
        super(MultiResolutionSTFT, self).__init__()
        assert len(fft_lengths) == len(frame_lengths) == len(frame_steps)
        self.scope = 'stft_loss' if scope is None else scope

        self.stft_losses = []
        for frame_length, frame_step, fft_length in zip(frame_lengths, frame_steps, fft_lengths):
            self.stft_losses.append(STFT(frame_length, frame_step, fft_length, scope='stft_'.format(frame_length)))

    def call(self, y, x):
        with tf.variable_scope(self.scope):
            sc_loss = 0.0
            mag_loss = 0.0

            for f in self.stft_losses:
                sl, mag_l = f.call(y, x)
                sc_loss += tf.reduce_mean(sl)
                mag_loss += tf.reduce_mean(mag_l)

            sc_loss /= len(self.stft_losses)
            mag_loss /= len(self.stft_losses)

            return sc_loss, mag_loss
