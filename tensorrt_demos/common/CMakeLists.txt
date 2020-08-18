cmake_minimum_required(VERSION 3.5)
project(common)

# find_package(CUDA 11.0 REQUIRED)
# find_package(CUDA 10.0 REQUIRED)
# enable_language("CUDA")

include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDA ${_CUDA_VERSION} REQUIRED)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(SOURCE_FILES 
	"decl_spec.h"
	"datatypes.h"
	"datatypes.cpp"
	"common.h"
	"common.cpp"
	"logging.h"
	)

add_library(common SHARED ${SOURCE_FILES})

target_include_directories(common INTERFACE ${TENSOR_RT_INCLUDE} ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common PUBLIC Eigen3::Eigen )
target_link_libraries(common PUBLIC ${CUDA_LIBRARIES})
