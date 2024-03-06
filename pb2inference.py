# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: 
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
import argparse
from datasets import audio

def pb2inference(args):
    tf.reset_default_graph()
    my_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.pb, 'rb') as fid:
        serialized_graph = fid.read()
        my_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(my_graph_def, name = '')

    my_graph = tf.get_default_graph()

    inputs = my_graph.get_tensor_by_name('mel_G:0')

    audio_out = my_graph.get_tensor_by_name('MelGAN/Generator/Tanh:0')

    with tf.Session(graph=my_graph) as sess:
        mel = np.load(args.file)
        mel = np.expand_dims(mel, axis=0)

        feed_dict = {inputs: mel}

        out = sess.run(audio_out, feed_dict=feed_dict)
        out = np.squeeze(out, 0)
        out = np.squeeze(out, 1)
        audio.save_wav(out, args.out, sr=16000)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-p',
        '--pb',
        type=str,
        default='/home/bairong/pb/melgan.pb')
    parser.add_argument(
        '-f', 
        '--file',
        default='/home/bairong/mel-tron/test.npy')
    parser.add_argument(
        '-o',
        '--out',
        default='/home/bairong/melgan/1.wav')
    args = parser.parse_args()

    pb2inference(args)


if __name__ == '__main__':
    main()

