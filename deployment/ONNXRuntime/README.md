## FreeYOLO ONNXRuntime

### Download FreeYOLO ONNX file
Main results on COCO-val:

| Model          |  Scale  |    AP    |    AP50    |  ONNX(opset=11)  |  ONNX(opset=10)  |
|----------------|---------|----------|------------|------------------|------------------|
| FreeYOLO-Nano  |  640    |   30.5   |   50.3     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_opset_11.onnx) | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_opset_10.onnx) |
| FreeYOLO-Tiny  |  640    |   34.4   |   53.9     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_opset_11.onnx) | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_opset_10.onnx) |
| FreeYOLO-Large |  640    |   48.3   |   68.5     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_large_opset_11.onnx) | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_large_opset_10.onnx) |
| FreeYOLO-Huge  |  640    |   50.0   |   69.5     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_huge_opset_11.onnx) | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_huge_opset_10.onnx) |


### Convert Your Model to ONNX

First, you should move to <FreeYOLO_HOME> by:
```shell
cd <FreeYOLO_HOME>
cd tools/
```
Then, you can:

1. Convert a standard FreeYOLO model by:
```shell
python3 export_onnx.py --output-name yolo_free_large.onnx -v yolo_free_large --weight ../weight/coco/yolo_free_large/yolo_free_large.pth --no_decode -nc 80 --img_size 640
```

Notes:
* -n: specify a model name. The model name must be one of the [yolox-s,m,l,x and yolox-nano, yolox-tiny, yolov3]
* -c: the model you have trained
* -o: opset version, default 11. **However, if you will further convert your onnx model to [OpenVINO](https://github.com/Megvii-BaseDetection/YOLOX/demo/OpenVINO/), please specify the opset version to 10.**
* --no-onnxsim: disable onnxsim
* To customize an input shape for onnx model,  modify the following code in tools/export_onnx.py:

    ```python
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)
    ```

### ONNXRuntime Demo

Step1.
```shell
cd <YOLOX_HOME>/deployment/ONNXRuntime
```

Step2. 
```shell
python3 onnx_inference.py --weight ../../weights/onnx/11/yolo_free_large.onnx -i ../test_image.jpg -s 0.3 --img_size 640
```
Notes:
* --weight: your converted onnx model
* -i: input_image
* -s: score threshold for visualization.
* --img_size: should be consistent with the shape you used for onnx convertion.
