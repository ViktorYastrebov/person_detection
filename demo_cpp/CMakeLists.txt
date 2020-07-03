cmake_minimum_required(VERSION 3.5)
project(video_detection_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

# INFO: set directory to the base of built OpenCV
if (${BUILD_TYPE} STREQUAL "debug")
	# set(INSTALLED_OPENCV_PATH "f:/developer02/workspace/deps/opencv_master_sdk8.1_debug")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_debug")
else()
	# set(INSTALLED_OPENCV_PATH "f:/developer02/workspace/deps/opencv_master_sdk8.1_release")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_release")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
add_subdirectory(detection_engine)
add_subdirectory(single_demo)
add_subdirectory(multi_demo)
