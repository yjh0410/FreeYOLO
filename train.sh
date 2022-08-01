# Train FreeYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_free_v1 \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        # --resume weights/coco/yolo_free_v2/yolo_free_v2_epoch_51_37.24.pth

# # Train YOLOF
# python train.py \
#         --cuda \
#         -d coco \
#         --root /mnt/share/ssd2/dataset/ \
#         -v yolof \
#         --ema \
#         --fp16 \
#         --eval_epoch 10 \
