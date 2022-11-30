#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import logging as log
import os
import time
import sys
sys.path.append('../../../')

import cv2
import numpy as np

from openvino.inference_engine import IECore

from utils.pre_process import PreProcessor
from utils.post_process import PostProcessor
from utils.vis_tools import visualize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("openvino inference sample")
    parser.add_argument("-m", "--model", type=str, default="../../../weights/onnx/10/yolo_free_large.xml",
                        help="Input your XML model.")
    parser.add_argument("-i", "--image_path", type=str, default='../../test_image.jpg',
                        help="Path to your input image.")
    parser.add_argument("-o", "--output_dir", type=str, default='../../../det_results/openvino/',
                        help="Path to your output directory.")
    parser.add_argument("-s", "--score_thr", type=float, default=0.35,
                        help="Score threshould to filter the result.")
    parser.add_argument('-d', '--device', default='CPU', type=str,
                        help='Optional. Specify the target device to infer on; CPU, GPU, \
                            MYRIAD, HDDL or HETERO: is acceptable. The sample will look \
                            for a suitable plugin for device specified. Default value \
                            is CPU.')
    parser.add_argument("-size", "--img_size", type=int, default=640,
                        help="Specify an input shape for inference.")

    return parser.parse_args()


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=args.model)

    if len(net.input_info) != 1:
        log.error('Sample supports only single input topologies')
        return -1
    if len(net.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'FP32'
    net.outputs[out_blob].precision = 'FP16'

    # Get a number of classes recognized by a model
    num_of_classes = max(net.outputs[out_blob].shape)

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # class color for better visualization
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]

    # preprocessor
    prepocess = PreProcessor(img_size=args.img_size)

    # postprocessor
    postprocess = PostProcessor(
        img_size=args.img_size, strides=[8, 16, 32],
        num_classes=80, conf_thresh=args.score_thr, nms_thresh=0.5)

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    input_shape = tuple([args.img_size, args.img_size])
    origin_img = cv2.imread(args.image_path)
    x, ratio = prepocess(origin_img)

    # ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    t0 = time.time()
    output = exec_net.infer(inputs={input_blob: x})
    print("inference time: {:.1f} ms".format((time.time() - t0)*1000))

    # ---------------------------Step 8. Process output--------------------------------------------------------------------
    output = output[out_blob]

    t0 = time.time()
    # post process
    bboxes, scores, labels = postprocess(output)
    bboxes /= ratio
    print("post-process time: {:.1f} ms".format((time.time() - t0)*1000))

    # ---------------------------Step 9. Visualization--------------------------------------------------------------------
    # visualize detection
    origin_img = visualize(
        img=origin_img,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        vis_thresh=args.score_thr,
        class_colors=class_colors
        )

    # show
    cv2.imshow('openvino detection', origin_img)
    cv2.waitKey(0)

    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)


if __name__ == '__main__':
    sys.exit(main())
