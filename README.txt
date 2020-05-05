### Used resources:

* https://github.com/YunYang1994/tensorflow-yolov3.git for tensorflow-yolov3 from scratch
* Fast RCNN/SSDlite Mobile/SSD Mobile nets are downloaded from official site of TF
* OpenCV must be built with GPU.
  Steps:
  1. wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
     wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
	 unzip opencv.zip
	 unzip opencv_contrib.zip
  2 building OpenCV by cmake:
	```
    cmake -D CMAKE_BUILD_TYPE=Release
		-D BUILD_opencv_world=ON
		-D CMAKE_INSTALL_PREFIX="d:\viktor_project\installs\opencv_4.2_gpu"
		-D WITH_CUDA=ON
		-D OPENCV_DNN_CUDA=ON
		-D ENABLE_FAST_MATH=1
		-D CUDA_FAST_MATH=1
		-D CUDA_ARCH_BIN=7.0
		-D WITH_CUBLAS=1
		-D OPENCV_EXTRA_MODULES_PATH="d:\viktor_project\installs\opencv_contrib-4.2.0\modules"
		-D HAVE_opencv_python3=ON
		-D PYTHON_EXECUTABLE="C:\tools\miniconda3\envs\demo_env\python.exe"
		-D BUILD_EXAMPLES=ON
		-G"Visual Studio 15 2017 Win64" ..
	```
	
* C++ yolo3 library example:
  Source: https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420
  Steps:
  1. wget https://pjreddie.com/media/files/yolov3.weights
  2. for names: https://github.com/pjreddie/darknet/blob/master/data/coco.names
  3. 