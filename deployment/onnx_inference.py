#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime

from dataset.transforms import ValTransforms
from utils.post_process import PostProcessor
from utils.vis_tools import visualize


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument("-v", "--version", type=str, default="yolo_free.onnx",
                        help="Input your onnx model.")
    parser.add_argument("-i", "--image_path", type=str, default='test_image.jpg',
                        help="Path to your input image.")
    parser.add_argument("-o", "--output_dir", type=str, default='det_results/onnx/',
                        help="Path to your output directory.")
    parser.add_argument("-s", "--score_thr", type=float, default=0.3,
                        help="Score threshould to filter the result.")
    parser.add_argument("-size", "--img_size", type=int, default=640,
                        help="Specify an input shape for inference.")
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    # read an image
    input_shape = tuple(args.img_size, args.img_size)
    origin_img = cv2.imread(args.image_path)

    # preprocess
    # TODO: preprocess image
    image = None
    x = None

    # inference
    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: x[None, :, :, :]}
    output = session.run(None, ort_inputs)

    # post process
    # TODO: post process

    # visualize detection
    # TODO: vis

    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
