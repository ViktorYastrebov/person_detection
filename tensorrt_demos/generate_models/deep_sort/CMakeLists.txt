cmake_minimum_required(VERSION 2.6)

project(deep_sort_builder)

# USAGE EXAMPLE:
# cmake -DCMAKE_BUILD_TYPE=Release -DTENSOR_RT_BASE="d:/viktor_project/installs/TensorRT-7.1.3.4.Windows10.x86_64.cuda-11.0.cudnn8.0/TensorRT-7.1.3.4" -G"Visual Studio 15 2017 Win64" ..

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/output")

# find_package(CUDA 11.0 REQUIRED)
find_package(CUDA 10.0 REQUIRED)

# set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
	
	set(TENSOR_RT_INCLUDE ${TENSOR_RT_BASE}/include)
	include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE})
	link_directories(${CUDA_LIBRARIES} ${TENSOR_RT_BASE}/lib)
endif()

# find_package(OpenCV PATHS ${OPENCV_MODULE_PATH} NO_DEFAULT_PATH)
# include_directories(OpenCV_INCLUDE_DIRS)

set(SOURCE_FILES "main.cpp" "utils.h" "logging.h")

add_executable(deep_sort_builder ${SOURCE_FILES})

target_link_libraries(deep_sort_builder nvinfer)
target_link_libraries(deep_sort_builder nvonnxparser)
 