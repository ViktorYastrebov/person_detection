#pragma once 

#ifdef ganz_camera_engine_EXPORTS
   #define GANZ_CAMERA_ENGINE_DECL __declspec(dllexport)
#else
   #define GANZ_CAMERA_ENGINE_DECL __declspec(dllimport)
#endif