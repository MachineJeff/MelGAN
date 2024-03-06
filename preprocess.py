# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: preprocess data 
'''
import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import extract

def run_preprocess(args):
    in_dir = os.path.join(args.base_dir, args.dataset)
    out_dir = os.path.join(args.base_dir, args.output)

    mel_dir = os.path.join(out_dir, 'mels')
    audio_dir = os.path.join(out_dir, 'audios')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    metadata = extract.build_from_path(in_dir, mel_dir, audio_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'melgan.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')

    print("preprocess data done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', default='/data/tts/yichao.li/jj')
    parser.add_argument('--dataset', default='Baker-5000')
    parser.add_argument('--output', default='melgan_train')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()

    run_preprocess(args)

