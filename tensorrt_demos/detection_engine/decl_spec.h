#pragma once 

#if WIN32
  #ifdef detection_engine_EXPORTS
     #define ENGINE_DECL __declspec(dllexport)
  #else
     #define ENGINE_DECL __declspec(dllimport)
  #endif
#else
   #define ENGINE_DECL
#endif

