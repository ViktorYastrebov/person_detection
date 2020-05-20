cmake_minimum_required(VERSION 3.5)
project(ganz_camera_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")

# Manual paths to SDK dirs, libs etc
set(CAMERA_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sdk)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CAMERA_SDK_DIR}/include)

add_executable(server_demo "main.cpp")

#target_link_libraries(server_demo ${CAMERA_SDK_DIR}/windows/x64/sdk.lib)
#target_link_libraries(server_demo ${CAMERA_SDK_DIR}/windows/x86/sdk.lib)


