cmake_minimum_required(VERSION 3.5)

project(yolov3_builder)

# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# find_package(CUDA ${_CUDA_VERSION} REQUIRED)

# set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
# set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

#option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE})
link_directories(${CUDA_LIBRARIES} ${TENSOR_RT_BASE}/lib)

find_package(OpenCV PATHS ${OPENCV_MODULE_PATH} NO_DEFAULT_PATH)
# include_directories(OpenCV_INCLUDE_DIRS ${ROOT_DIR})

add_executable(yolov3_builder ${PROJECT_SOURCE_DIR}/yolov3-spp.cpp)

# message(STATUS "CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}")

target_include_directories(yolov3_builder PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${ROOT_DIR}
        )

target_link_libraries(yolov3_builder
	PRIVATE yolov3_plugin
	PRIVATE ${OpenCV_LIBS}
	)

