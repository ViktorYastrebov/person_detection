#pragma once 

#ifdef tracker_EXPORTS
   #define TRACKER_ENGINE __declspec(dllexport)
#else
   #define TRACKER_ENGINE __declspec(dllimport)
#endif