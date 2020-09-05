cmake_minimum_required(VERSION 3.5)
project(simple_demo)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCE_FILES "main.cpp")

find_package(CUDA ${_CUDA_VERSION} REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
link_directories(${TENSOR_RT_BASE}/lib)


add_executable(simple_demo ${SOURCE_FILES})
target_link_libraries(simple_demo detection_engine deep_sort_tracker sort_tracker inside_time_tracker)
