cmake_minimum_required(VERSION 3.5)
project(tensort_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BUILD_ARCH 64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(BUILD_ARCH 32)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")

set(TENSOR_RT_BASE "d:/viktor_project/installs/TensorRT-7.1.3.4.Windows10.x86_64.cuda-11.0.cudnn8.0/TensorRT-7.1.3.4")
set(TENSOR_RT_INCLUDE ${TENSOR_RT_BASE}/include)
set(TENSOR_RT_LIBS 	${TENSOR_RT_BASE}/lib/nvinfer.lib
					${TENSOR_RT_BASE}/lib/nvinfer_plugin.lib
					${TENSOR_RT_BASE}/lib/nvonnxparser.lib
					${TENSOR_RT_BASE}/lib/nvparsers.lib
					${TENSOR_RT_BASE}/lib/myelin64_1.lib
					)

# INFO: set directory to the base of built OpenCV
if (${BUILD_TYPE} STREQUAL "debug")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.4/opencv_cuda_11_debug")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_debug")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_cuda_10.2_debug")
else()
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.4/opencv_cuda_11_release")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_release")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_cuda_10.2_release")
endif()


set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(detect_engine)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${DETECTION_ENGINE_INCLUDE})
set(SOURCE_FILES "main.cpp")
add_executable(tensort_demo ${SOURCE_FILES})
target_link_libraries(tensort_demo detection_engine)
