cmake_minimum_required(VERSION 3.5)
project(detection_engine)


find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)

set(DETECTION_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

set(SOURCE_FILES
		"yolov3_model.cpp"
		"yolov4_model.cpp"
		# INFO: just for Visual Studio
		"decl_spec.h"
		"device_utils.h"
		"base_model.cpp"
		"base_model.h"
		"yolov3_model.h"
		"yolov4_model.h"
		"yolov3_batch.h"
		"yolov3_batch.cpp"
		"yolov4_batch.h"
		"yolov4_batch.cpp"
		# JUST FOR TESTING, really not useful
		"yolo_tiny.h"
		"yolo_tiny.cpp"
		"yolov3_spp.h"
		"yolov3_spp.cpp"
		)

add_library(detection_engine SHARED ${SOURCE_FILES})

target_include_directories(detection_engine INTERFACE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(detection_engine ${OpenCV_LIBS})
