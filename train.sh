# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_nano \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        # --resume weights/coco/yolo_free_large/yolo_free_large_47.1.pth \
