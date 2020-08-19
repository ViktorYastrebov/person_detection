#pragma once 

//INFO: defined in CMake.txt
#ifdef yolov3_layer_export
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
