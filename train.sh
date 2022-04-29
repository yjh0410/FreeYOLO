python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolox_n \
        -m sim_ota \
        --ema \
        --eval_epoch 10
