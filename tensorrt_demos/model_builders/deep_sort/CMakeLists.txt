cmake_minimum_required(VERSION 3.5)

project(deep_sort_builder)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA ${_CUDA_VERSION} REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE} ${ROOT_DIR})
link_directories(${CUDA_LIBRARIES} ${TENSOR_RT_BASE}/lib)

set(SOURCE_FILES "main.cpp")

add_executable(deep_sort_builder ${SOURCE_FILES})

target_link_libraries(deep_sort_builder nvinfer)
target_link_libraries(deep_sort_builder nvonnxparser)
