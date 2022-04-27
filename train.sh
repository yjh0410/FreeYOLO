python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolox_t \
        -m sim_ota \
        --ema \
        --eval_epoch 10
