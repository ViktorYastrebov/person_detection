cmake_minimum_required(VERSION 3.5)
project(tracker)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCE_FILES
		"decl_spec.h"
		"kalman_tracker.h"
		"kalman_tracker.cpp"
		"trackers_pool.h"
		"trackers_pool.cpp"
		"Hungarian.h"
		"Hungarian.cpp"
	)

add_library(tracker SHARED ${SOURCE_FILES})

target_include_directories(tracker INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tracker ${OpenCV_LIBS})
