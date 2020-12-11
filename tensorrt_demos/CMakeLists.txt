cmake_minimum_required(VERSION 3.5)
project(tensort_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

if(WIN32)
    set(_CUDA_VERSION 10.0)
    set(_TBB_ROOT "d:/viktor_project/installs/tbb-2020.2-win")
    # set(_TBB_ROOT "f:/developer02/workspace/deps/tbb-2020.2-win")

    set(_TBB_INCLUDE_DIR ${_TBB_ROOT}/include)

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(BUILD_ARCH 64)
	    set(_TBB_LIB_DIR ${_TBB_ROOT}/lib/intel64/vc14)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(BUILD_ARCH 32)
	    set(_TBB_LIB_DIR ${_TBB_ROOT}/lib/ia32/vc14)
    endif()
else()
    set(_CUDA_VERSION 11.1)
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
if(WIN32)
    # INFO: might better to allow to set as cmake param
    set(TENSOR_RT_BASE "d:/viktor_project/installs/TensorRT-7.1.3.4.Windows10.x86_64.cuda-11.0.cudnn8.0/TensorRT-7.1.3.4")
    #set(TENSOR_RT_BASE "f:/developer02/workspace/deps/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11")
    set(Eigen3_DIR "d:/viktor_project/installs/eigen-3.3.7_build/share/eigen3/cmake")
    #set(Eigen3_DIR "f:/developer02/workspace/deps/eigen_3.3.7_installed/share/eigen3/cmake")
    set(Qt5_DIR "d:/viktor_project/installs/qt-5/5.15.0/msvc2019_64")
    #set(Qt5_DIR "f:/developer02/workspace/deps/qt-5/5.15.0/msvc2019_64")

    set(TENSOR_RT_INCLUDE ${TENSOR_RT_BASE}/include)

    # INFO: set directory to the base of built OpenCV
    if (${BUILD_TYPE} STREQUAL "debug")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.4/opencv_cuda_10_debug")
	# set(INSTALLED_OPENCV_PATH "f:/developer02/workspace/deps/opencv_4.4.0_cuda_10_debug")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_cuda_10.2_debug")
    else()
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.4/opencv_cuda_11_release")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.4/opencv_cuda_10_release")
	# set(INSTALLED_OPENCV_PATH "f:/developer02/workspace/deps/opencv_4.4.0_cuda_10_release")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_cuda_10.2_release")
    endif()
else()
    set(INSTALLED_OPENCV_PATH "/home/user/build_deps/opencv-4.5_ffmpeg")
    set(Eigen3_DIR "/home/user/build_deps/eigen-3.3.8")
endif()


set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(common)
add_subdirectory(detection_engine)
add_subdirectory(deep_sort_tracker)
add_subdirectory(sort_tracker)
add_subdirectory(model_builders)
if(WIN32)
    add_subdirectory(ui_demo)
endif(WIN32)

add_subdirectory(simple_demo)
if(UNIX)
    set(TRITON_CLIENT_BASE "/home/user/build_deps/triton-2.5/client")
    set(TRITON_CLIENT_INCLUDE ${TRITON_CLIENT_BASE}/include)
    set(TRITON_CLIENT_LIB_DIR ${TRITON_CLIENT_BASE}/lib)
    add_subdirectory(triton_clients)
endif()

