cmake_minimum_required(VERSION 3.5)
project(detection_engine)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCE_FILES
		"yolov3_model.cpp"
		# INFO: just for Visual Studio
		"decl_spec.h"
		"device_utils.h"
		"yolov3_model.h")

add_library(detection_engine SHARED ${SOURCE_FILES})

target_link_libraries(detection_engine ${OpenCV_LIBS})
