# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: 
'''
import numpy as np
import tensorflow as tf

hparams = tf.contrib.training.HParams(
    num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
    num_freq = 1025, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    rescale = True, #Whether to rescale audio prior to preprocessing
    rescaling_max = 0.999, #Rescaling value

    silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing

    n_fft = 2048, #Extra window size is filled with 0 paddings to match this parameter
    hop_size = 200, #For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    win_size = 800, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate = 16000, #22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
    frame_shift_ms = None, #Can replace hop_size parameter. (Recommended: 12.5)
    
    preemphasize = True, #whether to apply filter
    preemphasis = 0.97, #filter coefficient.

    signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
    symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, 
    
    min_level_db = -100,
    ref_level_db = 20,
    fmin = 95, #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax = 7600, #To be increased/reduced depending on data.
    )
