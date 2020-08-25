cmake_minimum_required(VERSION 3.5)
project(sort_tracker)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR})

set(SOURCE_FILES
		"decl_spec.h"
		"kalman_tracker.h"
		"kalman_tracker.cpp"
		"trackers_pool.h"
		"trackers_pool.cpp"
	)

add_library(sort_tracker SHARED ${SOURCE_FILES})

target_link_libraries(sort_tracker common)
target_link_libraries(sort_tracker ${OpenCV_LIBS})
