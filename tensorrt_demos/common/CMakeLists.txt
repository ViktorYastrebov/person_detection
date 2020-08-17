cmake_minimum_required(VERSION 3.5)
project(common)

# find_package(CUDA 11.0 REQUIRED)
find_package(CUDA 10.0 REQUIRED)
 
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

# enable_language("CUDA")

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70)
# set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++14;-g;-G;-gencode;arch=compute_30;code=sm_30)

# set_target_properties(detection_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# set(Eigen3_DIR "d:/viktor_project/installs/eigen-3.3.7_build/share/eigen3/cmake")

include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE} ${OpenCV_INCLUDE_DIRS} ${ROOT_DIR})
# link_directories(${CUDA_LIBRARIES} ${TENSOR_RT_BASE}/lib)
# link_directories(${CUDAToolkit_LIBRARY_DIR})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDA 10.0 REQUIRED)

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

# target_link_libraries(detection_engine PUBLIC cudart)
# target_link_libraries(detection_engine PUBLIC nvinfer)
# target_link_libraries(detection_engine PUBLIC ${OpenCV_LIBS})
