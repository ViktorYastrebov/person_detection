#pragma once 

#ifdef common_EXPORTS
#define COMMON_EXPORT __declspec(dllexport)
#else
#define COMMON_EXPORT __declspec(dllimport)
#endif