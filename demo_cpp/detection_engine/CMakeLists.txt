cmake_minimum_required(VERSION 3.5)
project(detection_engine)

add_subdirectory(detection)
add_subdirectory(tracker)
add_subdirectory(deep_sort_tracker)

set(TRACKER_ENGINE_INCLUDE ${TRACKER_ENGINE_INCLUDE} PARENT_SCOPE)
set(DETECTION_ENGINE_INCLUDE ${DETECTION_ENGINE_INCLUDE} PARENT_SCOPE)
set(DEEP_SORT_TRACKER_INCLUDE ${DEEP_SORT_TRACKER_INCLUDE} PARENT_SCOPE)
