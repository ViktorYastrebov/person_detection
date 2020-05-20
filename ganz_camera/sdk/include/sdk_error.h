#ifndef __SDK_ERROR_H__
#define __SDK_ERROR_H__

///< 错误码
typedef enum sdks_error_e
{
	SDKS_ERR_OK,
	SDKS_ERROR_UNKNOWN = 1,                        ///< 未知错误
	SDKS_ERROR_SERVER_BUSY = 2,                    ///< 服务器忙
	SDKS_ERROR_NOT_SUPPORT = 3,                    ///< 不支持
	SDKS_ERROR_PARAMETER = 4,                      ///< 参数错误
	SDKS_ERROR_REQUEST_TIMEOUT = 5,                ///< 请求超时
	SDKS_ERROR_DEV_AUTH_CODE = 3100,               ///< 授权码错误,授权失败
	SDKS_ERROR_DEV_BIND_TO_DEV = 3101,             ///< 设备已绑定到其它设备
	SDKS_ERROR_DEV_BIND_TO_ACCOUNT = 3102,         ///< 设备已绑定到账户
	SDKS_ERR_NO_FRAME = 10000,                    ///< render没有刷到视频帧
	SDKS_ERR_UNREACHABLE,		                  ///< 设备不可达
	SDKS_ERR_CONNECT_TIMEOUT,	                  ///< 连接设备超时
	SDKS_ERR_DISCONNECT,		                  ///< 连接断开
	SDKS_ERR_DECODE,			                  ///< 解码器开太多
	SDKS_ERR_REC,				                  ///< 录像开太多
	SDKS_ERR_CAPTURE,			                  ///< 抓图失败
	SDKS_ERR_VIEW,				                  ///< 开太多窗口
	SDKS_ERR_NOT_OPEN_VIEW,	                      ///< 没有打开窗口
	SDKS_ERR_TOO_MANY_STREAM,	                  ///< 打开的流太多
}sdks_error_e;

#endif
