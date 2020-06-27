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
	
	
	
Update:

1. There is used the latest OpenCV version from the master branch:
   OpenCV: download master and checkout to 
		commit c3e8a82c9c0b18432627fe100db332b42d64b8d3 (HEAD)
		Merge: d3aaf2d3a3 e58e545584
		Author: Alexander Alekhin <alexander.a.alekhin@gmail.com>
		Date:   Thu May 28 23:35:11 2020 +0000

		Merge remote-tracking branch 'upstream/3.4' into merge-3.4
		
	OpenCV contrib download master and checkout to:
		commit 8fc6067b5987316d0586bc5db45c10aadad8c97f (HEAD)
		Merge: a712bf79 2e6e9e9e
		Author: Alexander Alekhin <alexander.a.alekhin@gmail.com>
		Date:   Sat May 23 18:58:03 2020 +0000

		Merge pull request #2543 from vchiluka5:NVOF-Build-Error-Fix
		
		
	example:
		cmake -D CMAKE_BUILD_TYPE=Release
		-DCMAKE_SYSTEM_VERSION=8.1
		-D BUILD_opencv_world=ON
		-D CMAKE_INSTALL_PREFIX="f:\developer02\workspace\deps\opencv-4.3.0_release"
		-D WITH_CUDA=ON
		-D OPENCV_DNN_CUDA=ON
		-D ENABLE_FAST_MATH=1
		-D CUDA_FAST_MATH=1
		-D CUDA_ARCH_BIN=5.2
		-D WITH_CUBLAS=1
		-D OPENCV_EXTRA_MODULES_PATH="f:\developer02\workspace\deps\opencv-4.3.0_contrib\modules"
		-D BUILD_EXAMPLES=OFF
		-G"Visual Studio 15 2017 Win64" ..
		
2. Models files. All models weights & cfg files are located in model directory.
   It must be put to the root dir of executable file. Intenaly it relates to the current working dir + "models" + model_name
   You may change the file location and its paths to anything you want but it needs the code correction
   
3. There has been added the Tracker library which uses Kalman-Filter for trakcing the objects based on detections per frame

4. Demo program parameters:
      -n "Model Name" for now it could be YoloV3 or YoloV4
	  -f "path_to_the_video_file"
	  -c confidence threshold for detectorion model. If value is not set default will be used = 0.3
	  
   
   
	
	
