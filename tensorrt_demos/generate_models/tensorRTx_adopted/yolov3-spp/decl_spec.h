#pragma once 

#ifdef yololayer_export
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
