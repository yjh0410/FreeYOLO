# YOLOX
This project is my reproduce of **YOLOX**.

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yolox python=3.6
```

- Then, activate the environment:
```Shell
conda activate yolox
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
- [x] Simple OTA

# Main results on VOC 2007 test

| Model                 |  Scale   |   Matcher   |   mAP@0.5   | Weight |
|-----------------------|----------|-------------|-------------|--------|
| YOLOX-S               |  640     |    Basic    |     78.2    |    -   |
| YOLOX-S               |  640     |   SimOTA    |     |    -   |

`Basic` matcher is leveraged from `FCOS`, so it is a fixed label assignment method.

# Main results on COCO-val

| Model                 |  Scale   |   mAP   | Weight |
|-----------------------|----------|---------|--------|
| YOLOX-S               |  640     |         |    -   |
| YOLOX-M               |  640     |         |    -   |
| YOLOX-L               |  640     |         |    -   |
| YOLOX-X               |  640     |         |    -   |
| YOLOX-T               |  416     |         |    -   |
| YOLOX-N               |  416     |         |    -   |

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
               -v yolox_d53 \
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
               -v yolox_d53 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight \
               --show
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolox_d53 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight \
               --show
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v yolox_d53 \
               --img_size 640 \
               --cuda \
               --weight path/to/weight \
               --show
```
