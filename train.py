# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: train script
'''
import math
import os
import time
import traceback
import argparse
import tensorflow as tf

import numpy as np
from models.melgan import MelGAN
from datasets.feeder import Feeder
import infolog
from datasets import audio
log = infolog.log


def add_stats(model):
    with tf.variable_scope('stats') as scope:
        tf.summary.scalar('Generator_Loss', model.loss_g)
        tf.summary.scalar('Discriminator_Loss', model.loss_d)
        tf.summary.scalar('Learning Rate', model.learning_rate)

        return tf.summary.merge_all()


def train(args, log_dir):
    save_dir = os.path.join(log_dir, 'melgan_pretrained')
    tensorboard_dir = os.path.join(log_dir, 'tensorboard_events')
    wav_dir = os.path.join(log_dir, 'wav_of_melgan')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'melgan_model.ckpt')
    input_path = os.path.join(args.base_dir, args.input)
    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_path)
    log('Using model: %s' % args.model)
    
    tf.set_random_seed(1234)

    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    with tf.variable_scope('feeder') as scope:
        feeder = Feeder(coord, input_path)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('MelGAN') as scope:
        model = MelGAN(feeder.mel_G, feeder.audio_G)
        model.add_loss()
        model.add_optimizer(global_step)
        stats = add_stats(model)

    saver = tf.train.Saver(max_to_keep=10)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7

    with tf.Session(config = config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            try:
                checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                    log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
                    saver.restore(sess, checkpoint_state.model_checkpoint_path)
                else:
                    log('No model to load at {}'.format(save_dir), slack=True)
                    saver.save(sess, checkpoint_path, global_step=global_step)
            except:
                pass

            feeder.start_threads(sess)
            
            loss_g = 0.0
            loss_d = 0.0
            step = 0
            while not coord.should_stop():

                if step % 2 == 0:
                    t1 = time.time()
                    step, loss_g, _= sess.run([global_step, model.loss_g, model.optimize_g])

                    message = 'Step {:7d} [ {:.3f} sec/step,     Generator_Loss = {:.5f} ]'.format(step//2, time.time() - t1, loss_g)
                    log(message, slack=(step % args.checkpoint_interval == 0))

                else:
                    t1 = time.time()
                    step, loss_d, _    = sess.run([global_step, model.loss_d, model.optimize_d])

                    message = 'Step {:7d} [ {:.3f} sec/step, Discriminator_Loss = {:.5f} ]'.format(step//2, time.time() - t1, loss_d)
                    log(message, slack=(step % args.checkpoint_interval == 0))

                if math.isnan(loss_g) or math.isnan(loss_d):
                    log('Loss exploded to at step {}!'.format(step), slack=True)
                    raise Exception('Loss Exploded')

                if step % (2*args.summary_interval) == 0:
                    summary_writer.add_summary(sess.run(stats), step)

                if step % (2*args.checkpoint_interval) == 0:
                    log('Saving checkpoint to: {}-{}'.format(checkpoint_path, step//2))
                    saver.save(sess, checkpoint_path, global_step=step)

                    log('Saving audio...')
                    out1 = sess.run(model.G_fake_audio[0])
                    out1 = np.squeeze(out1, 1)
                    audio.save_wav(out1, os.path.join(wav_dir, 'step-{}-audio_1.wav'.format(step//2)), sr=16000)

                    out2 = sess.run(model.G_fake_audio[1])
                    out2 = np.squeeze(out2, 1)
                    audio.save_wav(out2, os.path.join(wav_dir, 'step-{}-audio_2.wav'.format(step//2)), sr=16000)

        except Exception as e:
            log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', default='/data/tts/yichao.li/jj')
    parser.add_argument('-l', '--log_dir', default='')

    parser.add_argument('--model', default='MelGAN')
    parser.add_argument('--input', default='melgan_train/melgan.txt')
    parser.add_argument('--restore_step', type=bool, default=True, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=40)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)

    log_dir = os.path.join(args.log_dir, 'logs-{}'.format(args.model))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'MelGAN_train_log'), args.model, args.slack_url)

    train(args, log_dir)
