#pragma once 

#if WIN32
   #ifdef deep_sort_tracker_EXPORTS
       #define DEEP_SORT_TRACKER __declspec(dllexport)
   #else
       #define DEEP_SORT_TRACKER __declspec(dllimport)
   #endif
#else
   #define DEEP_SORT_TRACKER
#endif


