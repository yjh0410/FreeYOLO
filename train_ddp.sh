# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yolo_free_large \
                                                    --ema \
                                                    --fp16 \
                                                    --eval_epoch 10
