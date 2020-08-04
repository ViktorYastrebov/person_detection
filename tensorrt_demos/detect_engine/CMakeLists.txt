cmake_minimum_required(VERSION 3.5)
project(detection_engine)

set(DETECTION_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

find_package(CUDA 11.0 REQUIRED)
find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)
 
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

enable_language("CUDA")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70)
# set_target_properties(detection_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include_directories(
	${TENSOR_RT_INCLUDE}
	${CUDA_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${ROOT_DIR}
)

set(SOURCE_FILES 
	"decl_spec.h"
	"generic_detector.h"
	"generic_detector.cpp"
	"detection_node_gpu.cu"
	)

# cuda_add_library(detection_engine SHARED ${SOURCE_FILES})
add_library(detection_engine SHARED ${SOURCE_FILES})

target_include_directories(detection_engine INTERFACE ${TENSOR_RT_INCLUDE} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(detection_engine "${TENSOR_RT_LIBS}" "${CUDA_LIBRARIES}" ${OpenCV_LIBS})