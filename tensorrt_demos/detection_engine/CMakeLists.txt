cmake_minimum_required(VERSION 3.5)
project(detection_engine)

set(DETECTION_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

find_package(CUDA ${_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)
 
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

link_directories(${CUDAToolkit_LIBRARY_DIR}
	#       	${TENSOR_RT_BASE}/lib
	)

set(SOURCE_FILES 
    "decl_spec.h"
    "base_model.h"
    "base_model.cpp"
    "common_model.cpp"
    "common_model.h"
    "yolov3_model.h"
    "yolov3_model.cpp"
    "yolov5_model.h"
    "yolov5_model.cpp"
    "builder.h"
    "builder.cpp"
    )

add_library(detection_engine SHARED ${SOURCE_FILES})

target_include_directories(detection_engine INTERFACE 
	#	${TENSOR_RT_INCLUDE}
	${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(detection_engine  PRIVATE yolov3_plugin
                                               yolov5_plugin
                                               common
                                        PUBLIC  ${CUDA_LIBRARIES}
                                                nvinfer
                                                ${OpenCV_LIBS}
                     )
