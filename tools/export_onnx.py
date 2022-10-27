#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Thanks to YOLOX: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/export_onnx.py

import argparse
import os
from loguru import logger

import torch
from torch import nn

from models.basic.conv import SiLU
from utils.misc import load_weight, replace_module
from config import build_config
from models import build_model


def make_parser():
    parser = argparse.ArgumentParser("FreeYOLO onnx deploy")
    # basic
    parser.add_argument("--output-name", type=str, default="yolo_free.onnx",
                        help="output name of models")
    parser.add_argument("--input", default="images", type=str,
                        help="input node name of onnx model")
    parser.add_argument("--output", default="weights/onnx/", type=str,
                        help="output node name of onnx model")
    parser.add_argument("-o", "--opset", default=11, type=int,
                        help="onnx opset version")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--dynamic", action="store_true", default=False,
                        help="whether the input shape should be dynamic or not")
    parser.add_argument("--no-onnxsim", action="store_true", default=False,
                        help="use onnxsim or not")
    parser.add_argument("-f", "--exp_file", default=None, type=str,
                        help="experiment description file")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str,
                        help="ckpt path")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument("--decode_in_inference", action="store_true", default=False,
                        help="decode in inference or not")

    # model
    parser.add_argument('-v', '--version', default='yolo_free', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--no_decode_in_inference", action="store_true", default=False,
                        help="not decode in inference or yes")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    device = torch.device('cpu')

    # config
    cfg = build_config(args)

    # build model
    model = build_model(args=args, 
                        cfg=cfg,
                        device=device, 
                        num_classes=80, 
                        trainable=False)

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)
    model = model.to(device).eval()

    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, cfg['test_size'], cfg['test_size'])

    torch.onnx._export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()