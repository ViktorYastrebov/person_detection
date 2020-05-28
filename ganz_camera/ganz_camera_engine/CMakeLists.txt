cmake_minimum_required(VERSION 3.5)
project(ganz_camera_engine)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)
add_definitions(-DSDKS_USE_DLL -D__RJ_WIN64__)

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

include_directories(${CMAKE_SOURCE_DIR}
					${CAMERA_SDK_INCLUDE}
					${FFMPEG_INCLUDE_DIR} 
					${OpenCV_INCLUDE_DIRS} 
					${_TBB_INCLUDE_DIR})
					
set(SOURCE_FILES "connection.cpp"
				 "connection.h"
				 "stream_data_holder.h"
				 "stream_data_holder.cpp"
				 "sdk_context.cpp"
				 "sdk_context.h"
				 "face_handler.cpp"
				 "face_handler.h"
				 "alarm_handler.cpp"
				 "alarm_handler.h"
				 "base_video_stream.h"
				 "base_video_stream.cpp"
				 "video_stream.cpp"
				 "video_stream.h"
				 "decoders/decoder.cpp"
				 "decoders/decoder.h"
				 "face_data.cpp"
				 "face_data.h"
				 "alarm_data.cpp"
				 "alarm_data.h"
				 "history_recoder.cpp"
				 "history_recoder.h"
				 "decl_spec.h"
				 "${CMAKE_SOURCE_DIR}/json/cJSON.h"
				 "${CMAKE_SOURCE_DIR}/json/cJSON.cpp"
	)

add_library(ganz_camera_engine SHARED ${SOURCE_FILES})
							
target_link_libraries(ganz_camera_engine 
						${CAMERA_SDK_LIB_PATH}/sdk.lib 
						${FFMPEG_LIBS}
						${OpenCV_LIBS}
						${_TBB_LIBS})
						
target_include_directories(ganz_camera_engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${_TBB_INCLUDE_DIR} ${CAMERA_SDK_INCLUDE})						