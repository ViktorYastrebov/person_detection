cmake_minimum_required(VERSION 3.5)
project(person_detection_cpp)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE) 

if (${BUILD_TYPE} STREQUAL "debug")
	set(_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu_debug")
else()
	set(_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu")
endif() 

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/argparser)

find_package(OpenCV REQUIRED PATHS ${_OPENCV_PATH} NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

# TODO: remove header files, it was done only for simplify VS navigation
add_executable(demo_cpp "main.cpp" "model_producer.cpp" "model.cpp" "model_producer.h" "model.h" "device_utils.h")

target_link_libraries(demo_cpp ${OpenCV_LIBS})

set_target_properties(demo_cpp PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)