cmake_minimum_required(VERSION 2.6)

project(yolov5)

# add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/output")

find_package(CUDA 11.0 REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    # include_directories(/usr/local/cuda/include)
    # link_directories(/usr/local/cuda/lib64)
	set(TENSOR_RT_INCLUDE ${TENSOR_RT_BASE}/include)
	include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE})
	link_directories(${CUDA_LIBRARIES} ${TENSOR_RT_BASE}/lib)
endif()


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/yololayer.h ${PROJECT_SOURCE_DIR}/decl_spec.h)
# INFO: WinNT only
target_compile_definitions(yololayer PUBLIC yololayer_export)

find_package(OpenCV PATHS ${OPENCV_MODULE_PATH} NO_DEFAULT_PATH)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 yololayer)
target_link_libraries(yolov5 ${OpenCV_LIBS})

# add_definitions(-O2 -pthread)

