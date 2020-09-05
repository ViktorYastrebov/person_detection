cmake_minimum_required(VERSION 3.5)
project(inside_time_tracker)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH) 

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR})

set(SOURCE_FILES
		"decl_spec.h"
		"in_area_tracker.h"
		"in_area_tracker.cpp"
		"in_area_track.h"
		"in_area_track.cpp"
	)

add_library(inside_time_tracker SHARED ${SOURCE_FILES})
target_link_libraries(inside_time_tracker PUBLIC deep_sort_tracker)
target_link_libraries(inside_time_tracker PUBLIC ${OpenCV_LIBS})
