cmake_minimum_required(VERSION 3.5)
project(ganz_camera_server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

# INFO: set directory to the base of built OpenCV
# if (${BUILD_TYPE} STREQUAL "debug")
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu_debug")
	
#	set(INSTALLED_OPENCV_PATH "F:/sdks/building/opencv/opencv-4.2/opencv-4.2.0_build_dbg_opencl")
#else()
	# set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_4.2_gpu")
#	set(INSTALLED_OPENCV_PATH "F:/sdks/building/opencv/opencv-4.2/opencv-4.2.0_build_rel_opencl")
#endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
#add_subdirectory(detection_engine)

# As far as I can see libevent does not work correct with cmakes.
# Have to make it manualy

# https://kukuruku.co/post/lightweight-http-server-in-less-than-40-lines-on-libevent-and-c-11/

set(LIBEVENT_BASE_DIR "F:/sdks/vc17/libevent-2.1.11")
set(LIBEVENT_INCLUDE_DIR ${LIBEVENT_BASE_DIR}/include)
set(LIBEVENT_LIBRARIES ${LIBEVENT_BASE_DIR}/lib/event_core.lib ${LIBEVENT_BASE_DIR}/lib/event.lib ${LIBEVENT_BASE_DIR}/lib/event_extra.lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${LIBEVENT_INCLUDE_DIR})

add_executable(server_demo "main.cpp" "server/server.h" "server/server.cpp")

target_link_libraries(server_demo ${LIBEVENT_LIBRARIES})
if(WIN32)
  target_link_libraries(server_demo wsock32 ws2_32)
endif()

