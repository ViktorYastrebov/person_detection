cmake_minimum_required(VERSION 3.5)
project(triton_client)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCE_FILES
	"main.cpp"
        "utils.cpp"
        "utils.h"
	)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)


# find_package(CUDA ${_CUDA_VERSION} REQUIRED)
# set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# link_directories(${TENSOR_RT_BASE}/lib)


add_executable(triton_client ${SOURCE_FILES})

target_include_directories(triton_client 
	PRIVATE ${TRITON_CLIENT_INCLUDE}
	PRIVATE ${OpenCV_INCLUDE_DIRS}
	PRIVATE ${ROOT_DIR}
	)

target_link_directories(triton_client PRIVATE ${TRITON_CLIENT_LIB_DIR})

target_link_libraries(triton_client PRIVATE 
	${OpenCV_LIBS}
	PRIVATE common
	PRIVATE httpclient
	)

