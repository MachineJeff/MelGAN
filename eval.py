# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: 
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import argparse
from synthesize import Synthesizer
from datasets import audio


def run(args):
    syn = Synthesizer()
    syn.load(args.checkpoint)

    for i in range(5):
        syn.synthesize(args.mel)
    out = syn.synthesize(args.mel)
    audio.save_wav(out, args.path, sr=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mel', default='/data/yichao.li/data/2/jj_origin/melgan_train/mels/mel-001004.npy')
    parser.add_argument('-c', '--checkpoint', default='/data/yichao.li/log/melgan/logs-MelGAN/melgan_pretrained/melgan_model.ckpt-580000')
    parser.add_argument('-p', '--path', default='1.wav')
    args = parser.parse_args()

    run(args)
