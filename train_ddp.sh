# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
python -m torch.distributed.run --nproc_per_node=4 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /data/datasets/ \
                                                    -v yolo_free_huge \
                                                    --ema \
                                                    --fp16 \
                                                    --eval_epoch 10
