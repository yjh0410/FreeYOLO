# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
python -m torch.distributed.run --nproc_per_node=4 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /data/datasets/ \
                                                    -v yolo_free_large \
                                                    -bs 16 \
                                                    -accu 1 \
                                                    --max_epoch 300 \
                                                    --wp_epoch 1 \
                                                    --eval_epoch 10 \
                                                    --num_workers 4 \
                                                    --ema \
                                                    --fp16 \
