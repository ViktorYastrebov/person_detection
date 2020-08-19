
## Detectors generation ##

All models are pretrained and can be downloaded from the 
* YoloV3 : https://github.com/ultralytics/yolov3 
* YoloV5s/m/l/x: https://github.com/ultralytics/yolov5
* https://github.com/ZQPei/deep_sort_pytorch.git you may find the model in the instructions

**Many thanks these guys !!! :)**

> Also downloaded models are placed in the `models/pytorch_models` folder

## How to build the TensoRT generator ##
1. For windows setup python environment with conda [Windows miniconda](https://docs.conda.io/en/latest/miniconda.html) and all requirements<br>
   Example:<br>
   `conda create --no-default-packages -n pytorch_env python==3.8`<br>
   `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`<br>
    > For more details please have a look at the [official site](https://pytorch.org)

   `conda activate pytorch_env`<br>
   `pip install -r requirements.txt`
    > This is an example for YoloV3-SPP, for some others could be different environment requirements.
   
2. Generate *.wts from pytorch implementation
	* YoloV3-SPP:<br>
       `git clone https://github.com/ultralytics/yolov3.git`<br>
       `cd yolov3`<br>
       Copy yolov3-spp-ultralytics.pt or yolov3-spp.pt to yolov3 folder: <br>
	   `cp ./models/pytorch_models/yolov3-spp-ultralytics.pt . //copy pt file to yolov3 folder`<br>
	   Copy gen_wts.py from `common/yolov3spp` to yolov3 folder: <br>
       `cp ../common/yolov3spp/gen_wts.py .`<br>
	   Generate weight file: <br>
       `python gen_wts.py yolov3-spp-ultralytics.pt //It should generate yolov3-spp_ultralytics68.wts file`<br>
       Build the project and copy weights file to the output folder <br>
	   Run yolov3_builder: `yolov3_builder.exe -s "path_to_wts_file"`<br>
	   There should be generated the yolov3-spp.engine
	* YoloV5:<br>
	   `git clone https://github.com/ultralytics/yolov5.git`<br>
	   `cd yolov5`
	   Copy gen_wts.py from `common/yolov5` to yolov5 folder: <br>
	   `cp ../common/yolov5/gen_wts.py .` <br>
	   Copy yolov5m/s/l/x.pt files to yolov5 folder: <br>
	   `cp ../models/pytorch_models/yolov5m.pt .`
	   Generate weights file: <br>
	   `python gen_wts.py --model "yolov5m.pt"`
	   Build the project and copy wieghts file to the output folder <br>
	   Run yolov5_builder: `yolov5_builder.exe -s  "path_to_weight_file"`<br>
	   There should be generated the yolov5\*.engine file
   
## Generate DeepSort model

1. Copy deep_sort_32.onnx exported model to the output directory <br>   
   and just run: `deep_sort_builder.exe deep_sort_32.onnx`
   
   > Note: If you need to generate the different batch size you should go to the<br> 
   > python_pytroch example from this repo and run ./deep_sort/model.py file. There you can set the batch size<br>  
   > by setting input shape for exporting: `x = torch.randn(32, 3, 128, 64)`

## Project dependecies ##

* TensorRT:<br>
     Tested with :
	 * TensorRT 7.0 with CUDA 10.0
	 * TensorRT-7.1.3.4 with CUDA 11.0
* OpenCV I think any version
* Eigen 3.3.7
* tbb-2020.2 (Needs only for UI demo)
* Qt 5.15 (Needs only for UI demo)


Really helpful links: 
* [YoloV3 TensoRT](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov3)
* [YoloV5 TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)


> <strong>Note</strong>: <em>Building engine must be done on the each new hardware</em>

 
