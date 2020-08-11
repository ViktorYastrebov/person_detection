# Generate Detectors #

All models are pretrained and can be downloaded from the 
* YoloV3 : https://github.com/ultralytics/yolov3 
* YoloV5s/m/l/x: https://github.com/ultralytics/yolov5

**Many thanks that guys !!! :)**  

> Also downloaded models are placed in the pytorch_models folder


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
       `//copy yolov3-spp.pt or yolov3.pt to yolov3 folder`<br>
       `cp ../tensorRTx_adopted/yolov3-spp/gen_wts.py . //copy from tensorRTx_adopted/yolov3-spp/gen_wts.py to yolov3 folder` <br>
       `cp ../pytorch_models/yolov3-spp-ultralytics.pt`
       `python gen_wts.py yolov3-spp-ultralytics.pt //It should generate yolov3-spp_ultralytics68.wts file`
       `// the master branch of yolov3 should work, if not, you can checkout cf7a4d31d37788023a9186a1a143a2dab0275ead`<br>
       `cp yolov3-spp_ultralytics68.wts ../tensorRTx_adopted/yolov3-spp/ //copy yolov3-spp_ultralytics68.wts file to yolov3-spp folder`
	* YoloV5:<br>
	   `git clone https://github.com/ultralytics/yolov5.git`<br>
	   `cd yolov5`
	   `//copy yolov5m/s/l/x.pt files to yolov5 folder`<br>
	   `cp ../tensorRTx_adopted/yolov3-spp/gen_wts.py`
	   `cp ../pytorch_models/yolov5m.pt .`
	   `python gen_wts.py --model "yolov5m.pt"`
	   `cp yolov5m.wts ../tensorRTx_adopted/yolov5/`
   
3. Build TensorRT engine
   * YoloV3-SPP:<br>
       `cd .\tensorRTx_adopted\yolov3-spp`
       `mkdir build`
       `cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_MODULE_PATH="d:/viktor_project/installs/opencv_4.4/opencv_cuda_11_release" -DTENSOR_RT_BASE="d:/viktor_project/installs/TensorRT-7.1.3.4.Windows10.x86_64.cuda-11.0.cudnn8.0/TensorRT-7.1.3.4" -G"Visual Studio 15 2017 Win64" ..`
        > Note: OpenCV must be build with the same version of the CUDA & cuDNN
    
        build the project(Release mode recommented)<br>
        `cd .\output\{BuildType} `<br>
        > do not forget to put the yolov3-spp_ultralytics68.wts to the output folder. Inside the exe file it has relative path like: ../yolov3-spp_ultralytics68.wts`
    
        Creates the TensorRT engine and serializes it to the file yolov3-spp.engine by following command:<br>
        `yolov3-spp.exe -s `<br>
	* YoloV5:<br>
       `cd .\tensorRTx_adopted\yolov5`
       `mkdir build`
       `cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_MODULE_PATH="d:/viktor_project/installs/opencv_4.4/opencv_cuda_11_release" -DTENSOR_RT_BASE="d:/viktor_project/installs/TensorRT-7.1.3.4.Windows10.x86_64.cuda-11.0.cudnn8.0/TensorRT-7.1.3.4" -G"Visual Studio 15 2017 Win64" ..`
        > Note: OpenCV must be build with the same version of the CUDA & cuDNN
    
        build the project(Release mode recommented)<br>
        `cd .\output\{BuildType} `<br>
        > do not forget to put the yolov5.wts to the output folder. Inside the exe file it has relative path like: ../yolov5m.wts`
    
        Creates the TensorRT engine and serializes it to the file yolov5.engine by following command:<br>
        `yolov3-spp.exe -s "model_name"`<br>

All this steps **must** be done on the target machine

Original links can be found in:
* [YoloV3 TensoRT](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov3)
* [YoloV5 TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)