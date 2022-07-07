python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_v2 \
        --ema \
        --fp16 \
        --eval_epoch 10
