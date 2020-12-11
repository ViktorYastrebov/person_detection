#pragma once 

#if WIN32
  #ifdef sort_tracker_EXPORTS
     #define TRACKER_ENGINE __declspec(dllexport)
  #else
     #define TRACKER_ENGINE __declspec(dllimport)
  #endif
#else
  #define TRACKER_ENGINE
#endif

