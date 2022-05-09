# FreeYOLO
Anchor-free YOLO detector.

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yolo python=3.6
```

- Then, activate the environment:
```Shell
conda activate yolo
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

We suggest that PyTorch should be higher than 1.9.0 and Torchvision should be higher than 0.10.3. 
At least, please make sure your torch is version 1.x.

# Tricks
- [x] Mosaic Augmentation
- [x] Mixup Augmentation
- [x] Multi scale training
- [x] Cosine Annealing Schedule
- [x] Training from Scratch
- [] Simple OTA

# Main results on COCO-val

| Model                 |  Scale   |   mAP   | Weight |
|-----------------------|----------|---------|--------|
| FreeYOLO-S            |  640     |         |    -   |
| FreeYOLO-M            |  640     |         |    -   |
| FreeYOLO-L            |  640     |         |    -   |
| FreeYOLO-X            |  640     |         |    -   |
| FreeYOLO-T            |  640     |         |    -   |
| FreeYOLO-N            |  640     |         |    -   |

# Train
## Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

## Multi GPUs
```Shell
sh train_ddp.sh
```

You can change the configurations of `train_ddp.sh`, according to your own situation.

# Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolo_s \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

# Demo
I have provide some images in `data/demo/images/`, so you can run following command to run a demo:

```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               -v yolo_s \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolo_s \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v yolo_s \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```
