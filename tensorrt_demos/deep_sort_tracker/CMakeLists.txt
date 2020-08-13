cmake_minimum_required(VERSION 3.5)
project(deep_sort_tracker)

# find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${DETECTION_ENGINE_INCLUDE} ${ROOT_DIR})

# set(DEEP_SORT_TRACKER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
# set(Eigen3_DIR "d:/viktor_project/installs/eigen-3.3.7_build/share/eigen3/cmake")
# find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(SOURCE_FILES
		"decl_spec.h"
#		"common.h"
#		"common.cpp"
#		"deep_sort.h"
#		"deep_sort.cpp"
		"hungarian_eigen/hungarian.h"
		"hungarian_eigen/hungarian.cpp"
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
