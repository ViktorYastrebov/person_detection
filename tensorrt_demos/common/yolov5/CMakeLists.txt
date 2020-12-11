cmake_minimum_required(VERSION 3.5)
project(yolov5_plugin)

# USAGE EXAMPLE:
# cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_MODULE_PATH="d:/viktor_project/installs/opencv_4.4/opencv_cuda_11_release" -DTENSOR_RT_BASE="d:/viktor_project/installs/TensorRT-7.1.3.4.Windows10.x86_64.cuda-11.0.cudnn8.0/TensorRT-7.1.3.4" -G"Visual Studio 15 2017 Win64" ..

find_package(CUDA ${_CUDA_VERSION} REQUIRED)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# include_directories(${CUDA_INCLUDE_DIRS}
	#	${TENSOR_RT_INCLUDE}
	#	)

# link_directories(${CUDA_LIBRARIES} ${TENSOR_RT_BASE}/lib)

cuda_add_library(yolov5_plugin SHARED ${PROJECT_SOURCE_DIR}/yolov5_layer.cu ${PROJECT_SOURCE_DIR}/yolov5_layer.h ${PROJECT_SOURCE_DIR}/decl_spec.h)
# INFO: WinNT only
if(WIN32)
	target_compile_definitions(yolov5_plugin PUBLIC yolov5_layer_export)
endif()

target_include_directories(yolov5_plugin INTERFACE ${CUDA_INCLUDE_DIRS})

target_link_libraries(yolov5_plugin nvinfer)
if(WIN32)
	target_link_libraries(yolov5_plugin ${CUDA_LIBRARIES})
endif()

