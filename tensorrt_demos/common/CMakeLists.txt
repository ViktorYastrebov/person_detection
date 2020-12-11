cmake_minimum_required(VERSION 3.5)
project(common)

add_subdirectory(yolov3spp)
add_subdirectory(yolov5)

# include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE} ${ROOT_DIR})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDA ${_CUDA_VERSION} REQUIRED)


set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

# include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(SOURCE_FILES 
    "decl_spec.h"
    "datatypes.h"
    "datatypes.cpp"
    "common.h"
    "common.cpp"
    "logging.h"
    "hungarian_eigen/hungarian.h"
    "hungarian_eigen/hungarian.cpp"
    "cost_matrix_type.h"
    )

add_library(common SHARED ${SOURCE_FILES})

message(STATUS "CUDA_INCLUDE_DIRS =${CUDA_INCLUDE_DIRS}")

target_include_directories(common PRIVATE ${CUDA_INCLUDE_DIRS}
				  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
				  PRIVATE ${ROOT_DIR}
				)

target_link_libraries(common PUBLIC Eigen3::Eigen )
target_link_libraries(common PUBLIC ${CUDA_LIBRARIES})

