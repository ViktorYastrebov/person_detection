#pragma once 

//INFO: defined in CMake.txt
#ifdef yolov5_layer_export
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
