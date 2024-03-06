# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2020-05-18
@Description: 
'''
import tensorflow as tf
import argparse
import os
from hparams import hparams
from synthesize import Synthesizer

def freeze2pb(args):
    synth = Synthesizer()
    synth.load(args.checkpoint)
    sess = synth.session
    output_path = os.path.join(args.output_dir, 'melgan.pb')


    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=args.node_name.split(","))

    for node in output_graph_def.node:
        if node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                node.input[0] = node.input[1]
                del node.input[1]

    with tf.gfile.GFile(output_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("freeze to pb done in {}".format(output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default='/home/bairong/log/melgan/logs-MelGAN/melgan_pretrained/melgan_model.ckpt-2024000')
    parser.add_argument(
        "-n",
        "--node_name",
        type=str,
        default= 'MelGAN/Generator/Tanh')
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='/home/bairong/pb')

    args = parser.parse_args()
    freeze2pb(args)

if __name__ == '__main__':
  main()
