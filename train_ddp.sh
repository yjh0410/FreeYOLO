# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yolo_free_large \
                                                    -bs 16 \
                                                    --max_epoch 300 \
                                                    --wp_epoch 1 \
                                                    --eval_epoch 10 \
                                                    --ema \
                                                    --fp16 \
