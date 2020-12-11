cmake_minimum_required(VERSION 3.5)
project(deep_sort_tracker)

find_package(CUDA ${_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED PATHS ${INSTALLED_OPENCV_PATH} NO_DEFAULT_PATH) 

link_directories(${CUDAToolkit_LIBRARY_DIR}
	#	${TENSOR_RT_BASE}/lib
	)

set(SOURCE_FILES
        "decl_spec.h"
        "linear_assignment.h"
        "linear_assignment.cpp"
        "tracker_impl.h"
        "tracker_impl.cpp"
        "tracker.h"
        "tracker.cpp"
        "track.h"
        "track.cpp"
        "kalman_filter.h"
        "kalman_filter.cpp"
        "nn_matching.h"
        "nn_matching.cpp"
        "base_tracker.cpp"
        "base_tracker.h"
        "time_tracker.cpp"
        "time_tracker.h"
        "deep_sort_types.cpp"
        "deep_sort_types.h"
        "time_track.cpp"
        "time_track.h"
        "time_tracker_impl.cpp"
        "time_tracker_impl.h"
        "deep_sort_types.cpp"
        "deep_sort_types.h"
        "deep_sort.cpp"
        "deep_sort.h"
    )

target_include_directories(common INTERFACE
	#       	${TENSOR_RT_INCLUDE}
       	${CUDA_INCLUDE_DIRS}
       	${OpenCV_INCLUDE_DIRS}
       	${CMAKE_CURRENT_SOURCE_DIR}
       	${ROOT_DIR}
	)

add_library(deep_sort_tracker SHARED ${SOURCE_FILES})

target_link_libraries(deep_sort_tracker PUBLIC  common
                                                nvinfer
                                        PRIVATE ${OpenCV_LIBS}
                     )

