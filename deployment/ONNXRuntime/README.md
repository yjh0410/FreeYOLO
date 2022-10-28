## FreeYOLO ONNXRuntime

### Convert Your Model to ONNX

First, you should move to <FreeYOLO_HOME> by:
```shell
cd <FreeYOLO_HOME>
cd tools/
```
Then, you can:

1. Convert a standard FreeYOLO model by:
```shell
python3 export_onnx.py --output-name yolo_free.onnx -n yolo_free --weight yolo_free.pth --no_decode
```

Notes:
* -n: specify a model name. The model name must be one of the [yolox-s,m,l,x and yolox-nano, yolox-tiny, yolov3]
* -c: the model you have trained
* -o: opset version, default 11. **However, if you will further convert your onnx model to [OpenVINO](https://github.com/Megvii-BaseDetection/YOLOX/demo/OpenVINO/), please specify the opset version to 10.**
* --no-onnxsim: disable onnxsim
* To customize an input shape for onnx model,  modify the following code in tools/export_onnx.py:

    ```python
    dummy_input = torch.randn(args.batch_size, 3, cfg['test_size'], cfg['test_size'])
    ```

