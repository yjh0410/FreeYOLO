# Train FreeYOLO
python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_huge \
        -bs 8 \
        -accu 8 \
        --max_epoch 300 \
        --wp_epoch 1 \
        --eval_epoch 10 \
        --ema \
        --fp16 \
        # --coco_pretrained weights/coco/yolo_free_large/yolo_free_large_47.1.pth \
        # --resume weights/coco/yolo_free_large/yolo_free_large_47.1.pth \
