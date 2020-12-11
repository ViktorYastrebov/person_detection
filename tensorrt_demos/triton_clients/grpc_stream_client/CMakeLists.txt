cmake_minimum_required(VERSION 3.5)
project(triton_grpc_stream_client)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCE_FILES
	"main.cpp"
        "utils.cpp"
        "utils.h"
#        "shm_utils.h"
#        "shm_utils.cpp"
        "cuda_buffer.h"
        "cuda_buffer.cpp"
	)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)


find_package(CUDA ${_CUDA_VERSION} REQUIRED)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

link_directories(${CUDAToolkit_LIBRARY_DIR})

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} 
	PRIVATE ${TRITON_CLIENT_INCLUDE}
	PRIVATE ${OpenCV_INCLUDE_DIRS}
	PRIVATE ${ROOT_DIR}
	PRIVATE ${CUDA_INCLUDE_DIRS}
	)

target_link_directories(${PROJECT_NAME}
                        PRIVATE ${TRITON_CLIENT_LIB_DIR}
                        PRIVATE ${CUDAToolkit_LIBRARY_DIR}
                       )

target_link_libraries(${PROJECT_NAME} PRIVATE 
	${OpenCV_LIBS}
	PRIVATE common
	PRIVATE grpcclient
	PRIVATE ${CUDA_LIBRARIES}
	)

