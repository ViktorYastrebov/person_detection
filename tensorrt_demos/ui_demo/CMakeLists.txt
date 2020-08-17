cmake_minimum_required(VERSION 3.5)
project(ui_demo)

find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED PATHS ${Qt5_DIR} NO_DEFAULT_PATH)

set(SOURCE_FILES
		"display_window.h"
		"display_window.cpp"
		"main_window.h"
		"main_window.cpp"
		"main.cpp"
   )

find_package(CUDA 10.0 REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

include_directories(${_TBB_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${TENSOR_RT_BASE}/lib ${_TBB_LIB_DIR})

add_executable(ui_demo WIN32 ${SOURCE_FILES})

target_link_libraries(ui_demo detection_engine deep_sort_tracker)
target_link_libraries(ui_demo ${Qt5Widgets_LIBRARIES})
target_link_libraries(ui_demo ${Qt5Gui_LIBRARIES})
target_link_libraries(ui_demo ${Qt5Core_LIBRARIES})
target_link_libraries(ui_demo tbb)
