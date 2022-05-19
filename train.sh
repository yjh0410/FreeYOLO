python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -v free_yolo_csp_d53 \
        --ema \
        --fp16 \
        --eval_epoch 10
