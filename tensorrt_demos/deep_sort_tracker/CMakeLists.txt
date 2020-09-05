cmake_minimum_required(VERSION 3.5)
project(deep_sort_tracker)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${DETECTION_ENGINE_INCLUDE} ${ROOT_DIR})

set(SOURCE_FILES
		"decl_spec.h"
		"linear_assignment.h"
		"linear_assignment.cpp"
		"tracker.h"
		"tracker.cpp"
		"track.h"
		"track.cpp"
		"kalman_filter.h"
		"kalman_filter.cpp"
		"nn_matching.h"
		"nn_matching.cpp"
	)

add_library(deep_sort_tracker SHARED ${SOURCE_FILES})
target_link_libraries(deep_sort_tracker common)
