cmake_minimum_required(VERSION 3.5)
project(multi_demo)
		  

find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

set(SOURCE_FILES
		"display_window.h"
		"display_window.cpp"
		"main_window.h"
		"main_window.cpp"
		"model_builder.h"
		"model_builder.cpp"
		"track_processor.h"
		"track_processor.cpp"
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

