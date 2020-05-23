#ifndef __RJ_TYPE_H__
#define __RJ_TYPE_H__

/*
	8,16,32,64 基础数据定义

	__RJ_WIN32__,		__RJ_WIN64__,			//x86
	__RJ_LINUX32__,		__RJ_LINUX64__,			//x86
	__RJ_LINUX32_ARM__,	__RJ_LINUX64_ARM__,		//arm
*/
#if defined(__RJ_WIN32__) || defined(__RJ_WIN64__)
	#ifndef __RJ_WIN__
		#define __RJ_WIN__
	#endif
#elif defined(__RJ_LINUX32__) || defined(__RJ_LINUX64__) || defined(__RJ_LINUX32_ARM__)
	#ifndef __RJ_LINUX__	
		#define __RJ_LINUX__
	#endif
#else
	#error Not Suport Platform
#endif

#if defined(__RJ_WIN32__) || defined(__RJ_LINUX32__) || defined(__RJ_LINUX32_ARM__)
	typedef unsigned short		RJ_BOOL;
	typedef signed char			int8;
	typedef unsigned char		uint8;
	typedef short				int16;
	typedef unsigned short		uint16;
	typedef int					int32;
	typedef unsigned int		uint32;
	typedef long long			int64;
	typedef unsigned long long	uint64;

	typedef int					int_t;		//整型,随机器字长变化
	typedef unsigned int		uint_t;		//无符号整型,随机器字长变化
	typedef long				long_t;		//整型,随机器字长变化
	typedef unsigned long		ulong_t;	//整型,随机器字长变化
#elif defined(__RJ_WIN64__) || defined(__RJ_LINUX64__) || defined(__RJ_LINUX64_ARM__)
	typedef unsigned short		RJ_BOOL;
	typedef char				int8;
	typedef unsigned char		uint8;
	typedef short				int16;
	typedef unsigned short		uint16;
	typedef int					int32;
	typedef unsigned int		uint32;
	typedef long long			int64;
	typedef unsigned long long	uint64;

	typedef long long			int_t;		//整型,随机器字长变化
	typedef unsigned long long	uint_t;		//无符号整型,随机器字长变化
	typedef long long			long_t;		//整型,随机器字长变化
	typedef unsigned long long	ulong_t;	//整型,随机器字长变化
#else
	#error No Define Platform
#endif

#if defined(__RJ_WIN__)
	#if defined(RJ_BUILD_DLL)
		#define RJ_API extern "C" __declspec(dllexport)
	#elif defined(RJ_USE_DLL)
		#define RJ_API extern "C"  __declspec(dllimport)
	#else
		#define RJ_API
	#endif
#else
	#ifdef __cplusplus
		#define RJ_API extern "C"
	#else
		#define RJ_API extern
	#endif
#endif

#define RJ_FALSE	0
#define RJ_TRUE		1
#endif

//end
