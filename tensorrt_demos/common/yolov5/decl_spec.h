#pragma once 

//INFO: defined in CMake.txt
#if WIN32
   #ifdef yolov5_layer_export
      #define EXPORT __declspec(dllexport)
   #else
      #define EXPORT __declspec(dllimport)
   #endif
#else
   #define EXPORT
#endif

