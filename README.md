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
- [x] [Mosaic Augmentation](https://github.com/yjh0410/FreeYOLO/blob/master/dataset/transforms.py)
- [x] [Mixup Augmentation](https://github.com/yjh0410/FreeYOLO/blob/master/dataset/transforms.py)
- [x] Multi scale training
- [x] Cosine Annealing Schedule

# Training Configuration
|   Configuration         |                      |
|-------------------------|----------------------|
| Batch Size (bs)         | 16                   |
| Init Lr                 | 0.01/64 × bs         |
| Lr Scheduler            | Cos                  |
| Optimizer               | SGD                  |
| ImageNet Predtrained    | True                 |
| Multi Scale Train       | 320, 352, ..., 640   |
| Mosaic                  | True                 |
| Mixup                   | True                 |

# Network
## AnchorYOLO & FreeYOLO
- Backbone: [CSPDarkNet-53](https://github.com/yjh0410/FreeYOLO/blob/master/models/backbone/cspdarknet.py)
- Neck: [SPP](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/spp.py)
- FPN: [YoloPaFPN](https://github.com/yjh0410/FreeYOLO/blob/master/models/neck/yolopafpn.py)
- Head: [DecoupledHead](https://github.com/yjh0410/FreeYOLO/blob/master/models/head/decoupled_head.py)

# Experiments
## COCO

Main results on COCO-val:

| Model        |  Scale  | FPS<sup><br>2080ti |  FLOPs  |  Params |    AP    |    AP50    |  Weight  |
|--------------|---------|--------------------|---------|---------|----------|------------|----------|
| YOLOF        |  608    |  74                |  64.1 B |  33.0 M |   39.9   |    60.3    | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolof_39.9_60.3.pth) |
| AnchorYOLO   |  608    |  45                |  76.3 B |  62.0 M |   42.5   |    62.1    | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_anchor_42.5_62.1.pth) |
| FreeYOLO     |  608    |  51                |  76.0 B |  61.8 M | **43.7** |  **62.6**  | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_43.7_62.6.pth) |

AP results on COCO-val:

| Model        |  Scale  |  AP      |  AP50      |  AP75      |  APs      |  APm      |  APl      |
|--------------|---------|----------|------------|------------|-----------|-----------|-----------|
| YOLOF        |  608    |   39.9   |    60.3    |    42.5    |    22.9   |    44.6   |   52.8    |
| AnchorYOLO   |  608    |   42.5   |    62.1    |    45.1    |  **28.5** |    47.6   |   53.1    |
| FreeYOLO     |  608    | **43.7** |  **62.6**  |  **46.7**  |    28.0   |  **49.2** | **57.4**  |

## VOC

| Model        |  Scale  |  AP50 |  Weight  |
|--------------|---------|-------|----------|
| YOLOF        |  608    |  83.7 | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolof_83.7.pth) |
| AnchorYOLO   |  608    |  84.4 | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_anchor_84.4.pth) |
| FreeYOLO     |  608    |  84.9 | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_84.9.pth) |


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

**In the event of a training interruption**, you can pass `--resume` the latest training
weight path (`None` by default) to resume training. For example:

```Shell
python train.py \
        --cuda \
        -d coco \
        -v yolo_free \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        --resume weights/coco/yolo_free/yolo_free_epoch_151_39.24.pth
```

Then, training will continue from 151 epoch.

# Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolo_free \
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
               -v yolo_free \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolo_free \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v yolo_free \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```
