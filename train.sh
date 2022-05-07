python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -v yolox_s \
        -m ota \
        --ema \
        --fp16 \
        --eval_epoch 10
