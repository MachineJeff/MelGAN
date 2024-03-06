# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: Data Feeder 
'''
import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from infolog import log


_batches_per_group = 32
_pad = 0


class Feeder:
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, coordinator, metadata_filename):
        super(Feeder, self).__init__()
        self._coord = coordinator
        self._offset = 0

        # Load metadata:
        self._meldir = os.path.join(os.path.dirname(metadata_filename), 'mels')
        self._audiodir = os.path.join(os.path.dirname(metadata_filename), 'audios')

        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]

        with tf.device('/cpu:0'):
            self._placeholders = [
                tf.placeholder(tf.float32, [None, None, 80], 'mel_G'),
                tf.placeholder(tf.float32, [None, None, 1], 'audio_G')
            ]

            # Create queue for buffering data:
            queue = tf.FIFOQueue(8, [tf.float32, tf.float32], name='input_queue')
            self._enqueue_op = queue.enqueue(self._placeholders)
            self.mel_G, self.audio_G = queue.dequeue()

            self.mel_G.set_shape(self._placeholders[0].shape)
            self.audio_G.set_shape(self._placeholders[1].shape)
    
    def start_threads(self, session):
        self._session = session
        thread = threading.Thread(name='background', target=self._enqueue_next_group)
        thread.daemon = True #Thread will close when parent quits
        thread.start()

    def _enqueue_next_group(self):
        while not self._coord.should_stop():
            start = time.time()

            # Read a group of examples:
            n = 16 # 16 recommended
            examples = [self._get_next_example() for i in range(n * _batches_per_group)]

            # Bucket examples based on similar output sequence length for efficiency:
            examples.sort(key=lambda x: len(x[1][0]))
            batches = [examples[i:i+n] for i in range(0, len(examples), n)]
            random.shuffle(batches)

            log('Generated {} batches of {} groups in {:.3f} sec'.format(n, len(batches), time.time() - start))
            for batch in batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
                self._session.run(self._enqueue_op, feed_dict=feed_dict)


    def _get_next_example(self):
        '''Loads a single example from disk'''
        if self._offset >= len(self._metadata):
            self._offset = 0
            random.shuffle(self._metadata)

        meta_g = self._metadata[self._offset]
        self._offset += 1
        mel_G = np.load(os.path.join(self._meldir, meta_g[0]))
        audio_G = np.load(os.path.join(self._audiodir, meta_g[1]))
        audio_G = np.expand_dims(audio_G, axis=1)

        return (mel_G, audio_G)


    def _prepare_batch(self, batch):
        random.shuffle(batch)

        mel_g = self._prepare_mel([x[0] for x in batch])
        audio_g = self._prepare_audio([x[1] for x in batch])

        return (mel_g, audio_g)


    def _prepare_mel(self, inputs):
        max_length = max((len(t)) for t in inputs)
        return np.stack([self._pad_mel_target(t, max_length) for t in inputs])


    def _pad_mel_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


    def _prepare_audio(self, inputs):
        max_length = max((len(t)) for t in inputs)
        return np.stack([self._pad_audio_target(x, max_length) for x in inputs])

    def _pad_audio_target(self, t, length):    
        return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)
