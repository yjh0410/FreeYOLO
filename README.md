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
| Multi Scale Train       | True                 |
| Mosaic                  | True                 |
| Mixup                   | True                 |


# Experiments
## COCO

Main results on COCO-val:

| Model        |  Scale  | FPS<sup><br>2080ti |  FLOPs  |  Params |    AP    |    AP50    |  Weight  |
|--------------|---------|--------------------|---------|---------|----------|------------|----------|
| YOLOF        |  608    |  74                |  87.7 B |  48.3 M |   41.0   |    61.3    | [github]() |
| AnchorYOLO   |  608    |  45                |  76.3 B |  62.0 M |   43.4   |    63.6    | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_anchor_43.4_63.6.pth) |
| FreeYOLO     |  640    |  58                |  72.4 B |  44.1 M |   44.8   |    65.8    | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_44.8_65.8.pth) |


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
