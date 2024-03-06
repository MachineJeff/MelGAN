# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: 
'''
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from hparams import hparams
from datasets import audio


def build_from_path(in_dir, mel_dir, audio_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            basename = parts[0]
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % basename)
            futures.append(executor.submit(partial(_process_utterance, mel_dir, audio_dir, basename, wav_path, hparams)))
            index += 1
        return [future.result() for future in tqdm(futures) if future.result() is not None]

def _process_utterance(mel_dir, audio_dir, index, wav_path, hparams):
    try:
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError:
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
        return None
    
    out = wav

    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    pad = audio.librosa_pad_lr(out, hparams.n_fft, audio.get_hop_size(hparams))

    out = np.pad(out, pad, mode='reflect')

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    out = out[:mel_frames * audio.get_hop_size(hparams)]

    # Pre filename
    mel_filename = 'mel-{}.npy'.format(index)
    audio_filename = 'audio-{}.npy'.format(index)

    # Write the spectrogram and audio to disk
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(audio_dir, audio_filename), out.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example
    return (mel_filename, audio_filename)

