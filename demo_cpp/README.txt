### Yolo v3 usage with OpenCV

* OpenCV must be build with GPU support & cuDNN for using it with GPU
  Steps:
  1.  wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
      wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
	  unzip opencv.zip
	  unzip opencv_contrib.zip
	 
   2. Example of build:
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
		
	3. Set in the CMake INSTALLED_OPENCV_PATH variable to built OpenCV.
	   Attention: It's build config dependent, Debug or release

* YoloV3 class required the path to the model weights(yolov3.weights) + config file(yolov3.cfg)
  For testing use demo_cpp simple exe file with the arguments.
  Example:
     -f "d:\viktor_project\test_data\videos\Running - 294.mp4" -m "..\\models\\yolo_v3\\yolov3.weights" -c "..\\models\\yolo_v3\\yolov3.cfg"
	 
	 Where -f file to video file
	       -m model wieghts
		   -c model config
		   
    YoloV3 class provides the list of bounding boxes of detected persons
