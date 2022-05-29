python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        --start_epoch 0 \
