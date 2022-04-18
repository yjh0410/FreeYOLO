python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolox_d53 \
        --grad_clip_norm -1. \
        --ema
