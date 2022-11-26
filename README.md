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

My environment:
- PyTorch = 1.9.1
- Torchvision = 0.10.1

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
- Download COCO.
```Shell
cd <FreeYOLO_HOME>
cd dataset/scripts/
sh COCO2017.sh
```

- Check COCO
```Shell
cd <FreeYOLO_HOME>
cd dataset/
python coco.py
```

Main results on COCO-val:

| Model          |  Scale  | FPS<sup><br>2080ti |  FLOPs   |  Params  |    AP    |    AP50    |  Weight  |
|----------------|---------|--------------------|----------|----------|----------|------------|----------|
| FreeYOLO-Nano  |  416    |                    |   1.9 G  |  2.0 M   |   27.1   |   45.5     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_27.1.pth) |
| FreeYOLO-Tiny  |  416    |                    |   5.9 G  |  6.2 M   |   31.1   |   49.4     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_31.1.pth) |
| FreeYOLO-Large |  640    |  58                |  144.8 G |  44.1 M  |   47.0   |   67.4     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_large_47.0.pth) |
| FreeYOLO-Huge  |  640    |                    |  257.8 G |  78.9 M  |      |        |  |

New AP results and weight files are coming ...

## CrowdHuman
- Download [CrowdHuman](https://www.crowdhuman.org/).
```
CrowdHuman
|_ CrowdHuman_train01.zip
|_ CrowdHuman_train02.zip
|_ CrowdHuman_train03.zip
|_ CrowdHuman_val.zip
|_ annotation_train.odgt
|_ annotation_val.odgt
```

- Prepare CrowdHuman
```
CrowdHuman
|_ CrowdHuman_train
|  |_ Images
|     |_ 273271,1a0d6000b9e1f5b7.jpg
|     |_ ...
|_ CrowdHuman_val
|  |_ Images
|     |_ 273271,1b9330008da38cd6.jpg
|     |_ ...
|_ annotation_train.odgt
|_ annotation_val.odgt
```

- Convert CrowdHuman to COCO format.
```Shell
cd <FreeYOLO_HOME>
cd tools/
python convert_crowdhuman_to_coco.py
```

- Check CrowdHuman
```Shell
cd <FreeYOLO_HOME>
cd dataset/
python crowdhuman.py
```

Main results on CrowdHuman-val:

| Model          |  Scale  |    AP    |    AP50    |  Weight  |
|----------------|---------|----------|------------|----------|
| FreeYOLO-Nano  |  416    |      |        |  |
| FreeYOLO-Tiny  |  416    |      |        |  |
| FreeYOLO-Large |  640    |      |        |  |
| FreeYOLO-Huge  |  640    |      |        |  |

-
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
        -v yolo_free_large \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        --resume weights/coco/yolo_free_large/yolo_free_large_epoch_151_39.24.pth
```

Then, training will continue from 151 epoch.

# Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolo_free_large \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

# Evaluation
```Shell
python eval.py -d coco-val \
               --cuda \
               -v yolo_free_large \
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
               -v yolo_free_large \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolo_free_large \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v yolo_free_large \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

# Deployment
1. [ONNX export and an ONNXRuntime](./deployment/ONNXRuntime/)
2. [OpenVINO in C++ and Python](./deployment/OpenVINO)