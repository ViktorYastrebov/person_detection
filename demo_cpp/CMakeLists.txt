cmake_minimum_required(VERSION 3.5)
project(person_detection_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

# INFO: set directory to the base of built OpenCV
if (${BUILD_TYPE} STREQUAL "debug")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu_debug")
	
	set(INSTALLED_OPENCV_PATH "F:/sdks/building/opencv/opencv-4.2/opencv-4.2.0_build_dbg_opencl")
else()
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu")
	set(INSTALLED_OPENCV_PATH "F:/sdks/building/opencv/opencv-4.2/opencv-4.2.0_build_rel_opencl")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
add_subdirectory(detection_engine)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_cpp "main.cpp")

target_link_libraries(demo_cpp  detection_engine)