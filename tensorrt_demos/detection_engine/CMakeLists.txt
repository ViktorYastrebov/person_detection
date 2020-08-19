cmake_minimum_required(VERSION 3.5)
project(detection_engine)

set(DETECTION_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

find_package(CUDA ${_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)
 
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR})
link_directories(${CUDAToolkit_LIBRARY_DIR} ${TENSOR_RT_BASE}/lib)

set(SOURCE_FILES 
	"decl_spec.h"
	"base_model.h"
	"base_model.cpp"
	"yolov3_model.h"
	"yolov3_model.cpp"
	"yolov5_model.h"
	"yolov5_model.cpp"
	"deep_sort.cpp"
	"deep_sort.h"
	)

add_library(detection_engine SHARED ${SOURCE_FILES})

target_include_directories(detection_engine INTERFACE ${TENSOR_RT_INCLUDE} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(detection_engine PRIVATE yolov3_plugin)
target_link_libraries(detection_engine PRIVATE yolov5_plugin)
target_link_libraries(detection_engine PRIVATE common)
target_link_libraries(detection_engine PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(detection_engine PUBLIC nvinfer)
target_link_libraries(detection_engine PUBLIC ${OpenCV_LIBS})

