python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof \
        --ema \
        --fp16 \
        --eval_epoch 10
