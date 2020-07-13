cmake_minimum_required(VERSION 3.5)
project(video_detection_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

set(CMAKE_PREFIX_PATH "d:/viktor_project/installs/qt-5/5.15.0/msvc2019_64")
# set(CMAKE_PREFIX_PATH "f:/developer02/workspace/deps/qt-5/5.15.0/msvc2019_64")
set(Qt5_DIR "d:/viktor_project/installs/qt-5/5.15.0/msvc2019_64")
# set(Qt5_DIR "f:/developer02/workspace/deps/qt-5/5.15.0/msvc2019_64")

set(_TBB_ROOT "d:/viktor_project/installs/tbb-2020.2-win")
# set(_TBB_ROOT "f:/developer02/workspace/deps/tbb-2020.2-win")

set(_TBB_INCLUDE_DIR ${_TBB_ROOT}/include)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BUILD_ARCH 64)
	set(_TBB_LIB_DIR ${_TBB_ROOT}/lib/intel64/vc14)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(BUILD_ARCH 32)
	set(_TBB_LIB_DIR ${_TBB_ROOT}/lib/ia32/vc14)
endif()


# INFO: set directory to the base of built OpenCV
if (${BUILD_TYPE} STREQUAL "debug")
	# set(INSTALLED_OPENCV_PATH "f:/developer02/workspace/deps/opencv_master_sdk8.1_debug")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_debug")
	
		set(_TBB_LIBS ${_TBB_LIB_DIR}/tbb_debug.lib
					  ${_TBB_LIB_DIR}/tbb_preview_debug.lib
					  ${_TBB_LIB_DIR}/tbbbind_debug.lib
					  )
	
else()
	# set(INSTALLED_OPENCV_PATH "f:/developer02/workspace/deps/opencv_master_sdk8.1_release")
	set(INSTALLED_OPENCV_PATH "d:/viktor_project/installs/opencv_master/opencv_release")
	set(_TBB_LIBS ${_TBB_LIB_DIR}/tbb.lib
					  ${_TBB_LIB_DIR}/tbb_preview.lib
					  ${_TBB_LIB_DIR}/tbbbind.lib
					  ) 				
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
add_subdirectory(detection_engine)
add_subdirectory(single_demo)
add_subdirectory(multi_demo)
