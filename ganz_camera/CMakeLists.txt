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

set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)

# Manual paths to SDK dirs, libs etc
set(CAMERA_SDK_DIR ${DEPS_DIR}\\sdk)
set(CAMERA_SDK_INCLUDE ${CAMERA_SDK_DIR}/include)
if(BUILD_ARCH EQUAL 64)
	set(CAMERA_SDK_LIB_PATH ${CAMERA_SDK_DIR}/windows/x64)
else()
	set(CAMERA_SDK_LIB_PATH ${CAMERA_SDK_DIR}/windows/x86)
endif()

# INFO: for some reason the all env vars are empty, TBB_INCLUDE_DIR etc
#       do it hard coded now :(
#set(TBB_DIR ${DEPS_DIR}/tbb-2020.2-win/cmake)
set(_TBB_ROOT ${DEPS_DIR}/tbb-2020.2-win)

set(FFMPEG_BASE_DIR ${DEPS_DIR}/ffmpeg-4.2.2-win64)
set(FFMPEG_INCLUDE_DIR ${FFMPEG_BASE_DIR}/include)
set(FFMPEG_LIB_DIR ${FFMPEG_BASE_DIR}/lib)

set(FFMPEG_LIBS ${FFMPEG_LIB_DIR}/avutil.lib
				${FFMPEG_LIB_DIR}/avformat.lib
				${FFMPEG_LIB_DIR}/avcodec.lib
				${FFMPEG_LIB_DIR}/swscale.lib
				)
				
set(_TBB_INCLUDE_DIR ${_TBB_ROOT}/include)
if(BUILD_ARCH EQUAL 64)
	set(_TBB_LIB_DIR ${_TBB_ROOT}/lib/intel64/vc14)
else()
	set(_TBB_LIB_DIR ${_TBB_ROOT}/lib/ia32/vc14)
endif()
				
# INFO: set directory to the base of built OpenCV
if (${BUILD_TYPE} STREQUAL "debug")
	# set(INSTALLED_OPENCV_PATH "F:/sdks/building/opencv/opencv-4.2/opencv-4.2.0_build_dbg_opencl")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_debug")
	# set(INSTALLED_OPENCV_PATH "F:/sdks/building/opencv/opencv-4.2/opencv-4.2.0_build_dbg")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu_debug")
	set(_TBB_LIBS ${_TBB_LIB_DIR}/tbb_debug.lib
				  ${_TBB_LIB_DIR}/tbb_preview_debug.lib
				  ${_TBB_LIB_DIR}/tbbbind_debug.lib
				  )
else()
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu")
	set(_TBB_LIBS ${_TBB_LIB_DIR}/tbb.lib
				  ${_TBB_LIB_DIR}/tbb_preview.lib
				  ${_TBB_LIB_DIR}/tbbbind.lib
				  )
endif()				
				
				
				
find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)
#find_package(TBB REQUIRED tbb)

add_definitions(-DSDKS_USE_DLL -D__RJ_WIN64__)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${CAMERA_SDK_INCLUDE}
					${FFMPEG_INCLUDE_DIR} 
					${OpenCV_INCLUDE_DIRS} 
					${_TBB_INCLUDE_DIR})
					
message(STATUS "Include dirs: ${CMAKE_CURRENT_SOURCE_DIR} ${CAMERA_SDK_INCLUDE} ${FFMPEG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${_TBB_INCLUDE_DIR}")

add_executable(server_demo 	"main.cpp"
							"camera_client/connection.cpp"
							"camera_client/connection.h"
							"camera_client/stream_data_holder.h"
							"camera_client/stream_data_holder.cpp"
							"camera_client/sdk_context.cpp"
							"camera_client/sdk_context.h"
							"camera_client/face_handler.cpp"
							"camera_client/face_handler.h"
							"camera_client/alarm_handler.cpp"
							"camera_client/alarm_handler.h"
							"camera_client/base_video_stream.h"
							"camera_client/base_video_stream.cpp"
							# "camera_client/video_stream_opencv.h"
							# "camera_client/video_stream_opencv.cpp"
							"camera_client/video_stream.cpp"
							"camera_client/video_stream.h"
							"camera_client/decoders/decoder.cpp"
							"camera_client/decoders/decoder.h"
							"camera_client/face_data.cpp"
							"camera_client/face_data.h"
							"camera_client/alarm_data.cpp"
							"camera_client/alarm_data.h"
							"camera_client/history_recoder.cpp"
							"camera_client/history_recoder.h"
							# Additional
							"json/cJSON.h"
							"json/cJSON.cpp"
							)

							
							
target_link_libraries( server_demo 
						${CAMERA_SDK_LIB_PATH}/sdk.lib 
						${FFMPEG_LIBS}
						${OpenCV_LIBS}
						${_TBB_LIBS})

# ADD_CUSTOM_COMMAND(TARGET server_demo POST_BUILD
#               COMMAND copy "${CAMERA_SDK_LIB_PATH}\\*.dll"
#               "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\\$<CONFIGURATION>"
#               COMMENT "Copying sdk.dll..."
#               )
