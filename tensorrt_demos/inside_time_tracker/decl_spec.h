#pragma once 

#ifdef inside_time_tracker_EXPORTS
   #define INSIDE_TIME_EXPORT __declspec(dllexport)
#else
   #define INSIDE_TIME_EXPORT __declspec(dllimport)
#endif