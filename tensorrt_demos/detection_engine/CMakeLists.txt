cmake_minimum_required(VERSION 3.5)
project(detection_engine)

set(DETECTION_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# find_package(CUDA 11.0 REQUIRED)
find_package(CUDA ${_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)
 
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# enable_language("CUDA")

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70)
# set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++14;-g;-G;-gencode;arch=compute_30;code=sm_30)

# set_target_properties(detection_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR})
link_directories(${CUDAToolkit_LIBRARY_DIR} ${TENSOR_RT_BASE}/lib)

set(SOURCE_FILES 
	"decl_spec.h"
	"generic_detector.h"
	"generic_detector.cpp"
	"Utils.h"
	"yololayer.cu"
	"yololayer.h"
	"deep_sort.cpp"
	"deep_sort.h"
	)

cuda_add_library(detection_engine SHARED ${SOURCE_FILES})
# add_library(detection_engine SHARED ${SOURCE_FILES})

target_include_directories(detection_engine INTERFACE ${TENSOR_RT_INCLUDE} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(detection_engine PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(detection_engine PUBLIC nvinfer)
target_link_libraries(detection_engine PUBLIC ${OpenCV_LIBS})
target_link_libraries(detection_engine PRIVATE common)
