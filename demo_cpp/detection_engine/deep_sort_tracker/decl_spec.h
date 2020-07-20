#pragma once 

#ifdef deep_sort_tracker_EXPORTS
   #define DEEP_SORT_TRACKER __declspec(dllexport)
#else
   #define DEEP_SORT_TRACKER __declspec(dllimport)
#endif