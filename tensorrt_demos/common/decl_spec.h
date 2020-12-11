#pragma once 

#if WIN32
  #ifdef common_EXPORTS
  #define COMMON_EXPORT __declspec(dllexport)
  #else
  #define COMMON_EXPORT __declspec(dllimport)
  #endif
#else
  #define COMMON_EXPORT
#endif

