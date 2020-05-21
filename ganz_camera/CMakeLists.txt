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

add_definitions(-DSDKS_USE_DLL)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CAMERA_SDK_INCLUDE})

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
							)

target_link_libraries( server_demo ${CAMERA_SDK_LIB_PATH}/sdk.lib)

# ADD_CUSTOM_COMMAND(TARGET server_demo POST_BUILD
#               COMMAND copy "${CAMERA_SDK_LIB_PATH}\\*.dll"
#               "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\\$<CONFIGURATION>"
#               COMMENT "Copying sdk.dll..."
#               )
