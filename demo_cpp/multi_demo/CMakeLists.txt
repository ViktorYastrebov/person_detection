cmake_minimum_required(VERSION 3.5)
project(multi_demo)

set(CMAKE_PREFIX_PATH "d:/viktor_project/installs/qt-5/5.15.0/msvc2019_64")

set(Qt5_DIR "d:/viktor_project/installs/qt-5/5.15.0/msvc2019_64")


set(_TBB_ROOT "d:/viktor_project/installs/tbb-2020.2-win")
set(_TBB_INCLUDE_DIR ${_TBB_ROOT}/include)

if(BUILD_ARCH EQUAL 64)
	set(_TBB_LIB_DIR ${_TBB_ROOT}/lib/intel64/vc14)
	message(STATUS "USING x64 TBB")
else()
	set(_TBB_LIB_DIR ${_TBB_ROOT}/lib/ia32/vc14)
	message(STATUS "USING x86 TBB")
endif()


if (${BUILD_TYPE} STREQUAL "debug")
	set(_TBB_LIBS ${_TBB_LIB_DIR}/tbb_debug.lib
					  ${_TBB_LIB_DIR}/tbb_preview_debug.lib
					  ${_TBB_LIB_DIR}/tbbbind_debug.lib
					  )
else()
	set(_TBB_LIBS ${_TBB_LIB_DIR}/tbb.lib
					  ${_TBB_LIB_DIR}/tbb_preview.lib
					  ${_TBB_LIB_DIR}/tbbbind.lib
					  ) 				
endif()					  

find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

set(SOURCE_FILES
		"display_window.h"
		"display_window.cpp"
		"main_window.h"
		"main_window.cpp"
		"model_builder.h"
		"model_builder.cpp"
		"main.cpp"
   )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${TRACKER_ENGINE_INCLUDE}
					${DETECTION_ENGINE_INCLUDE}
					${Qt5Core_INCLUDE_DIRS}
					${Qt5Widgets_INCLUDE_DIRS}
					${Qt5Gui_INCLUDE_DIRS}
					${_TBB_INCLUDE_DIR}
					)

add_executable(multi_demo WIN32 ${SOURCE_FILES})

target_link_libraries(multi_demo
					 tracker_engine
					 detection_engine
					 ${Qt5Widgets_LIBRARIES}
					 ${Qt5Gui_LIBRARIES}
					 ${Qt5Core_LIBRARIES}
					 ${_TBB_LIBS}
					 )

# qt5_use_package(multi_demo Widgets)
# qt5_use_package(multi_demo Core)
# qt5_use_package(multi_demo Gui)

