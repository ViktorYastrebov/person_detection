cmake_minimum_required(VERSION 3.5)

project(deep_sort_builder)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA ${_CUDA_VERSION} REQUIRED)


# include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE} ${ROOT_DIR})
# link_directories(${CUDA_LIBRARIES}
	#       	${TENSOR_RT_BASE}/lib
	#	)

set(SOURCE_FILES "main.cpp")

add_executable(deep_sort_builder ${SOURCE_FILES})

message(STATUS "CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}")

target_include_directories(deep_sort_builder
	PRIVATE ${CUDA_INCLUDE_DIRS}
	${ROOT_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
	)


target_link_libraries(deep_sort_builder PRIVATE ${CUDA_LIBRARIES} 
	PRIVATE nvinfer
	PRIVATE nvonnxparser
	)

