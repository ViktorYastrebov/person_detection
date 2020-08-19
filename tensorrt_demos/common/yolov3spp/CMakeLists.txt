cmake_minimum_required(VERSION 3.5)

project(yolov3_plugin)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA ${_CUDA_VERSION} REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSOR_RT_INCLUDE})
link_directories(${CUDA_LIBRARIES} ${TENSOR_RT_BASE}/lib)

cuda_add_library(yolov3_plugin SHARED ${PROJECT_SOURCE_DIR}/yolov3_spp_layer.cu ${PROJECT_SOURCE_DIR}/yolov3_spp_layer.h ${PROJECT_SOURCE_DIR}/decl_spec.h)
target_compile_definitions(yolov3_plugin PUBLIC yolov3_layer_export)
target_link_libraries(yolov3_plugin nvinfer)
target_link_libraries(yolov3_plugin ${CUDA_LIBRARIES})
 