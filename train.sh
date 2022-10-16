# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_anchor \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        # --resume weights/coco/yolo_free_v3/yolo_free_v3_epoch_201_43.21.pth
