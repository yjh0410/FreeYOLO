# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        --resume weights/coco/yolo_free/yolo_free_epoch_201_46.98.pth
