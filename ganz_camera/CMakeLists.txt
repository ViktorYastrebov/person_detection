cmake_minimum_required(VERSION 3.5)
project(ganz_camera_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BUILD_ARCH 64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(BUILD_ARCH 32)
endif()

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")

# Manual paths to SDK dirs, libs etc
set(CAMERA_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}\\sdk)
set(CAMERA_SDK_INCLUDE ${CAMERA_SDK_DIR}/include)
if(BUILD_ARCH EQUAL 64)
	set(CAMERA_SDK_LIB_PATH ${CAMERA_SDK_DIR}/windows/x64)
else()
	set(CAMERA_SDK_LIB_PATH ${CAMERA_SDK_DIR}/windows/x86)
endif()


set(FFMPEG_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg-4.2.2-win64")
set(FFMPEG_INCLUDE_DIR ${FFMPEG_BASE_DIR}/include)
set(FFMPEG_LIB_DIR ${FFMPEG_BASE_DIR}/lib)

set(FFMPEG_LIBS ${FFMPEG_LIB_DIR}/avutil.lib
				${FFMPEG_LIB_DIR}/avformat.lib
				${FFMPEG_LIB_DIR}/avcodec.lib
				${FFMPEG_LIB_DIR}/swscale.lib
				)
				
				
# INFO: set directory to the base of built OpenCV
if (${BUILD_TYPE} STREQUAL "debug")
	set(INSTALLED_OPENCV_PATH "F:/sdks/building/opencv/opencv-4.2/opencv-4.2.0_build_dbg_opencl")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu_debug")
else()
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu")
endif()				
				
find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)


add_definitions(-DSDKS_USE_DLL)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CAMERA_SDK_INCLUDE} ${FFMPEG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(server_demo 	"main.cpp"
							"camera_client/connection.cpp"
							"camera_client/connection.h"
							"camera_client/sdk_context.cpp"
							"camera_client/sdk_context.h"
							"camera_client/face_handler.cpp"
							"camera_client/face_handler.h"
							"camera_client/alarm_handler.cpp"
							"camera_client/alarm_handler.h"
							"camera_client/video_stream.cpp"
							"camera_client/video_stream.h"
							"camera_client/decoders/decoder.cpp"
							"camera_client/decoders/decoder.h"
							)

							
							
target_link_libraries( server_demo ${CAMERA_SDK_LIB_PATH}/sdk.lib ${FFMPEG_LIBS} ${OpenCV_LIBS})

# ADD_CUSTOM_COMMAND(TARGET server_demo POST_BUILD
#               COMMAND copy "${CAMERA_SDK_LIB_PATH}\\*.dll"
#               "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\\$<CONFIGURATION>"
#               COMMENT "Copying sdk.dll..."
#               )
