cmake_minimum_required(VERSION 3.5)
project(single_demo)

set(SOURCE_FILES
		"main.cpp"
		"utils.cpp"
		"utils.h"
   )

message(STATUS "TRACKER_ENGINE_INCLUDE = ${TRACKER_ENGINE_INCLUDE}")
message(STATUS "DETECTION_ENGINE_INCLUDE = ${DETECTION_ENGINE_INCLUDE}")
message(STATUS "DEEP_SORT_TRACKER_INCLUDE = ${DEEP_SORT_TRACKER_INCLUDE}")

#include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${TRACKER_ENGINE_INCLUDE} ${DETECTION_ENGINE_INCLUDE} "${CMAKE_CURRENT_SOURCE_DIR}/..")
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${DETECTION_ENGINE_INCLUDE} ${DEEP_SORT_TRACKER_INCLUDE} "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(single_demo ${SOURCE_FILES})

# target_link_libraries(single_demo tracker_engine detection_engine)
target_link_libraries(single_demo deep_sort_tracker detection_engine)
