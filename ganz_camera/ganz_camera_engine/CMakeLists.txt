cmake_minimum_required(VERSION 3.5)
project(ganz_camera_engine)

find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH)
add_definitions(-DSDKS_USE_DLL -D__RJ_WIN64__)

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

include_directories(${CMAKE_SOURCE_DIR}
					${CAMERA_SDK_INCLUDE}
					${OpenCV_INCLUDE_DIRS} 
					)
					
set(SOURCE_FILES "sdk_context.cpp"
				 "sdk_context.h"
				 "face_detector.cpp"
				 "face_detector.h"
				 "face_data.cpp"
				 "face_data.h"
				 "decl_spec.h"
				 "${CMAKE_SOURCE_DIR}/json/cJSON.h"
				 "${CMAKE_SOURCE_DIR}/json/cJSON.cpp"
	)

add_library(ganz_camera_engine SHARED ${SOURCE_FILES})
							
target_link_libraries(ganz_camera_engine 
						${CAMERA_SDK_LIB_PATH}/sdk.lib 
						${OpenCV_LIBS}
						)
						
target_include_directories(ganz_camera_engine INTERFACE
							${CMAKE_CURRENT_SOURCE_DIR}
							${OpenCV_INCLUDE_DIRS}
							${CAMERA_SDK_INCLUDE}
							)