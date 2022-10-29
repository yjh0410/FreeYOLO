# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_tiny \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        # --resume weights/coco/yolo_free_large/yolo_free_large_epoch_201_46.98.pth
