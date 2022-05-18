python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_s \
        --ema \
        --fp16 \
        --eval_epoch 10
