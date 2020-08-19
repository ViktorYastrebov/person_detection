cmake_minimum_required(VERSION 3.5)

project(yolov5_builder)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA ${_CUDA_VERSION} REQUIRED)

set(TENSOR_RT_INCLUDE ${TENSOR_RT_BASE}/include)
include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE})
link_directories(${CUDA_LIBRARIES} ${TENSOR_RT_BASE}/lib)

find_package(OpenCV PATHS ${OPENCV_MODULE_PATH} NO_DEFAULT_PATH)
include_directories(OpenCV_INCLUDE_DIRS ${ROOT_DIR})

add_executable(yolov5_builder ${PROJECT_SOURCE_DIR}/yolov5.cpp ${PROJECT_SOURCE_DIR}/common.hpp)
target_link_libraries(yolov5_builder yolov5_plugin)
target_link_libraries(yolov5_builder ${OpenCV_LIBS})
