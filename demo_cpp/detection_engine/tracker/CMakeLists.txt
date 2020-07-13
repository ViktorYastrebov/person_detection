cmake_minimum_required(VERSION 3.5)
project(tracker_engine)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${DETECTION_ENGINE_INCLUDE})

set(TRACKER_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

set(SOURCE_FILES
		"decl_spec.h"
		"kalman_tracker.h"
		"kalman_tracker.cpp"
		"trackers_pool.h"
		"trackers_pool.cpp"
		"Hungarian.h"
		"Hungarian.cpp"
		"deep_sort.h"
		"deep_sort.cpp"
	)

add_library(tracker_engine SHARED ${SOURCE_FILES})
target_link_libraries(tracker_engine ${OpenCV_LIBS})
