cmake_minimum_required(VERSION 3.5)
project(detection_engine)

add_subdirectory(detection)
add_subdirectory(tracker)

set(TRACKER_ENGINE_INCLUDE ${TRACKER_ENGINE_INCLUDE} PARENT_SCOPE)
set(DETECTION_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)