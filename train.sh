# Train FreeYOLO
python train.py \
        --cuda \
        -d widerface \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_tiny \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        --eval_first \
        --coco_pretrained weights/coco/yolo_free_tiny/yolo_free_tiny_31.1.pth \
        # --resume weights/coco/yolo_free_large/yolo_free_large_47.1.pth \
