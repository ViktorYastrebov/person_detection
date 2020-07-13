cmake_minimum_required(VERSION 3.5)
project(pytorch_check)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

# INFO: set directory to the base of built OpenCV
if (${BUILD_TYPE} STREQUAL "debug")
	# set(INSTALLED_OPENCV_PATH "f:/developer02/workspace/deps/opencv_master_sdk8.1_debug")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_debug")
	set(CMAKE_PREFIX_PATH "d:/viktor_project/installs/pytorch/debug/libtorch")
else()
	# set(INSTALLED_OPENCV_PATH "f:/developer02/workspace/deps/opencv_master_sdk8.1_release")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_release")
	set(CMAKE_PREFIX_PATH "d:/viktor_project/installs/pytorch/release/libtorch")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BUILD_ARCH 64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(BUILD_ARCH 32)
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")

set(SOURCE_FILES "main.cpp")



find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})

add_executable(pytorch_test ${SOURCE_FILES})

target_link_libraries(pytorch_test "${TORCH_LIBRARIES}")


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pytorch_test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug")
endif (MSVC)