python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free \
        --ema \
        --fp16 \
        --eval_epoch 10
