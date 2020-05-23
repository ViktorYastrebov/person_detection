#ifndef __SDK_DEF_H__
#define __SDK_DEF_H__
#include "sdk_error.h"

#if defined(WIN32)
#if defined(SDKS_BUILD_DLL)
#define SDKS_API extern "C" __declspec(dllexport)
#elif defined(SDKS_USE_DLL)
#define SDKS_API extern "C"  __declspec(dllimport)
#else
#define SDKS_API
#endif
#else
#ifdef __cplusplus
#define SDKS_API extern "C"
#else
#define SDKS_API extern
#endif
#endif

typedef void(*SDK_WIFI_CB)(unsigned int handle, char* p_data, void* p_obj);
typedef void(*SDK_ALARM_CB)(unsigned int handle, void** p_data, void* p_obj);
typedef void(*SDK_STREAM_CB)(unsigned int handle, int stream_id, void* p_data, void* p_obj);
typedef void(*SDK_DISCONN_CB)(unsigned int handle, void* p_obj);
typedef void(*SDK_CONNECT_CB)(unsigned int handle, void* p_obj);
typedef void(*SDK_PLAY_TIME_CB)(unsigned int handle, int stream_id, void* p_obj, const char* p_time);
typedef void(*SDK_STREAM_DATE_LEN)(unsigned long len);
typedef void(*SDK_DETECT_CB)(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);
typedef void(*SDK_FACE_CB)(unsigned int handle, int pic_type,void* p_data, int *data_len, void** p_result, void* p_obj);//pic_type: 1底库， 2实时库
typedef void(*SDK_MICROPHONE_CB)(unsigned int handle, void* p_data, int *data_len, void* p_obj);
typedef void(*SDK_INTERCOM_DB_CB)(unsigned int db, void*p_obj);

#define MAX_DEV_NAME_LEN  128
#define MAX_IP_BUF_LEN  128
#define CONST_MAX_ALARM_IO_NUM  10					//IO报警最大支持个数
#define CONST_MAX_PERIOD_RECORD_TIME_NUM  16				//允许的最大录像的时间段个数
#define CONST_MAX_ALARM_OUT_NUM   16						//允许的最大报警输出通道个数	
#define MAX_LENGTH_DEVICEID   32		//设备id长度
typedef enum  video_stream_type_e
{
	STREAM_TYPE_1 = 1,                     // HD
	STREAM_TYPE_2,                         // Smooth
	STREAM_TYPE_3
}video_stream_type_e;

enum Device_Type     //设备类型
{
	IPCAMERA = 1,	//网络摄像机设备
	DVR = 2,	//数字视频录像机设备
	DVS = 3,	//数字视频服务器设备
	IPDOME = 4,	//网络高速球
	NVR = 5,	//NVR
	ONVIF_DEVICE = 6,	//Onvif 设备
	DECODER = 7,	//解码器
	LPR = 8,	//车牌识别摄像机
	FISHEYE = 9,    //鱼眼设备类型。
	NVR_2 = 10,   //标示4.0的NVR
	IPP = 11,   //四目全景摄像机
	THERMAL_DEVICE = 13,	//热成像设备
	HUMAN_TEMPERATURE = 14,//人体测温仪
	FACE_DETECT = 15,//人脸检测
	VEHICLE_DETECT = 16,//国内车牌-智芯
	THERMAL_DOUBLE_SENSOR_DEVICE = 17,	//热成像双仓普通测温设备
	AIMULTI_OBJECT_DETECT = 18,//AI MULTI OBJECT DETECT
	THERMAL_DOUBLE_SENSOR_DOME_DEVICE = 19,	//热成像双仓普通测温设备(可见光接机芯)
	DOMECORE_THERMAL_DEVICE = 20,			//中安云台双ip接pmd1030机芯
	FACE_DEVICE = 21,  //人脸智能盒子
	HK_DVR = 100,	//HK DVR
	RS_DVR = 101,	//安联DVR
	DH_DVR = 102,	//大华DVR
	VIRTUAL_NVR = 103,   //用于兼容NVR客户端所使用的本地Server
	DOMECORE = 104	//机芯
};

typedef struct jy_facebase_info    //实时库
{
	int  key_id;		//4字节, 索引
	int  channel;    //4字节, 通道id
	int  similarity;	//4字节, 相似度
	long long time;		//8字节, 抓拍时间点，单位us
	char name[36];		//最多32字节,人脸名字,4位是结束符0
	char id[36];		//最多32字节,人脸工号
	char group[36];		//最多32字节,人脸所属库
	char type[36];		//最多32字节,人脸类型
	long long date;     //8字节, 人脸生日，单位us
	unsigned short gender;      //2字节,人脸性别		
}jy_facebase_info;

typedef struct jy_face_info       //底库
{
	int  key_id;				//索引
	int  expired;				//是否会过期（0从不过期，1已经过期，2未过期）
	char name[36];				//人脸名称
	char identity[36];			//人脸身份标识id（身份证或者工号）
	char group[36];				//人脸所属库
	char type[36];				//人脸类型（通常是人的职位信息，比如老师、学生）	
	long long   birthday;		//人脸出生日期
	long long	s_time;			//起始时间，单位为微妙
	long long	e_time;			//结束时间，单位为微妙
	unsigned short  gender;		//人脸性别
	unsigned short  similarity; //与底库的相似度
	float  temperature;			//人脸温度值
	//char padding[64];	//预留空间，64字节
}jy_face_info;

typedef struct tagFacePicData
{
	int                 pic_flag;					//1(底库)，2（实时库）
	char*				pszPicDate;					//数据(底库)
	unsigned long		nPicDataLength;				//数据有效长度(底库)
	jy_face_info        param;						//底库
	//char*				pszData;					//数据(实时库)
	//unsigned long		nDataLength;				//数据有效长度(实时库)
	jy_facebase_info    base_param;                //实时库
}tagFacePicData;

typedef struct FRAME_INFO {
	long nWidth;	// 画面宽，单位为像素，如果是音频数据则为0
	long nHeight;	// 画面高，单位为像素，如果是音频数据则为0
	long nStamp;	// 时标信息，单位毫秒
	long nType;		//数据类型，见下表
	long nFrameRate;// 编码时产生的图像帧率
}FRAME_INFO;

typedef struct tagAVFrameData
{
	long					nStreamFormat;						//1表示原始流，2表示TS混合流，3表示原始加密流，4表示PS混合流。
	long					nESStreamType;						//原始流类型，1表示视频，2表示音频。
	long					nEncoderType;						//编码格式。
	long					nCameraNo;							//摄像机号，表示数据来自第几路
	unsigned long			nSequenceId;						//数据帧序号
	long					nFrameType;							//数据帧类型,1表示I帧, 2表示P帧, 0表示未知类型
	long long				nAbsoluteTimeStamp;					//数据采集时间戳，单位为微妙
	long long				nRelativeTimeStamp;					//数据采集时间戳，单位为微妙
	char*					pszData;							//数据
	unsigned long			nDataLength;						//数据有效长度
	long					nFrameRate;							//帧率
	long					nBitRate;							//当前码率　
	long					nImageFormatId;						//当前格式
	long					nImageWidth;						//视频宽度
	long					nImageHeight;						//视频高度
	long					nVideoSystem;						//当前视频制式
	unsigned long			nFrameBufLen;						//当前缓冲长度

	long					nStreamId;							// 流ID
	long					nTimezone;							// 时区
	long					nDaylightSavingTime;				//夏令时
}ST_AVFrameData;

typedef struct tagTargeHead
{
	int					targe_id;				//目标Id
	unsigned short		type;					//类型：人脸，人体，车牌，车辆等
	unsigned short		x;						//起点横坐标
	unsigned short		y;						//起点纵坐标
	unsigned short		w;						//矩形宽度
	unsigned short		h;						//矩形高度
	unsigned short		attr_data_len;			//属性数据长度

}tagTargeHead;

typedef struct tagPersonFace
{
	//tagTargeHead		headInfo;
	unsigned short		confidence;				//置信度
	unsigned short		yaw;					//测角
	unsigned short		tilt;					//斜角
	unsigned short		pitch;					//仰角
	unsigned short		gender;					//性别
	unsigned short		age;					//年龄
	unsigned short		complexion;				//肤色
	unsigned short		minority;				//民族
	float				temperature;			//人脸温度
	char				reserve[32];			//预留字段
	unsigned int		land_mark_size;			//特征点个数
	//char*		        land_mark_data;			//特征点数据
}tagPersonFace;

typedef struct _upbody_info_s
{
	//tagTargeHead		headInfo;
	unsigned short		confidence;
	unsigned short		gender;//sex ,[0-100],100--woman
	unsigned short		age;
	unsigned short		complexion;
	unsigned char		backpack;
	unsigned char		human_move;
	unsigned char		move_direction;
	unsigned char		ride_bike;
	unsigned char		ride_motorbike;
	unsigned char		human_face_visible;
	unsigned char		human_facing;
	unsigned char		human_facing_confidence;
}upbody_info_s;

typedef struct _plate_info_s
{
	//tagTargeHead		headInfo;
	unsigned short		have_plate;//是否有车牌,0无,1有
	unsigned short		plate_angleH;//! 角度
	unsigned short		plate_angleV;//! 角度
	unsigned short		plate_color;	//! <车牌颜色
	unsigned short		plate_type;	//! <车牌类型
	unsigned short		plate_confidence; //! <车牌置信度
	unsigned short		plate_country;//! < 哪国车牌
	unsigned short		char_num;		//! <车牌字符个数
	char plate_num[32];		//! <车牌号
	char plate_char_confidence[32];//! <车牌号每个字符的置信度
}plate_info_s;

typedef struct _vehicle_info_s
{
	//tagTargeHead		headInfo;
	unsigned short		have_plate;//是否有车牌,0无,1有
	unsigned short		vehicle_plate_angleH;	//!<  车牌水平倾斜角度;
	unsigned short		vehicle_plate_angleV;	//!<  车牌垂直倾斜角度;
	unsigned short		vehicle_color;	//!<  车辆颜色;
	unsigned short		vehicle_type; //!<  车辆类型，机动车，非机动车;
	unsigned short		confidence;
	unsigned short		vehicle_model; //!<  车辆型号，大型汽车，小型汽车
	unsigned short		vehicle_speed;	//!<  车辆使用速度;
	unsigned short		Vehicle_moving;	//!< 车辆是否在移动
	unsigned short		char_num;		//! <车牌字符个数
	char				plate_num[32];		//! <车牌号
	char				plate_char_confidence[32];//! <车牌号每个字符的置信度
	unsigned short		Move_direction;//!< 车辆移动方向
	unsigned short		Move_direction_confidence;//!< 车辆移动方向可信度
	unsigned short		Vehicle_facing;//!< 车辆朝向
	unsigned short		Vehicle_facing_confidence;//!< 车辆朝向置信度
	char				trademark_utf8[32];//!< 车辆品牌
	unsigned short		trademark_utf8_confidence;//!< 车辆品牌置信度
	unsigned short		Reserve;
}vehicle_info_s;

typedef struct tagDetectFaceData
{
	char				magic[4];				//魔法号，默认0xffff ffff
	int					vesion;					//负载协议版本号，网络字节序
	unsigned int		total_len;				//整个结构长度，网络字节序
	unsigned int		picture_len;			//图片数据长度，网络字节序
	unsigned short		full_image_width;		//大图的宽，Full_corp=0有效，网络字节序
	unsigned short		full_image_hight;		//大图的高，Full_corp=0有效，网络字节序												
	//long long			capture_time;           //检测目标时刻单位秒，网络字节序
	unsigned int capture_timeh;
	unsigned int capture_timel;
	unsigned int		sequence_id;			//序号ID,，网络字节序
	char				full_crop;				//大图小图0（大）1（小）
	char				reserve[31];			//预留字段
	unsigned int		target_size;			//检测目标个数，网络字节序
}ST_DetectFaceData;
//热成像原始数据
typedef struct tagThermalAVData
{
	char				magic[4];				//魔法号，默认0xffff ffff
	unsigned int		vesion;					//负载协议版本号，网络字节序
	unsigned int		head_len;				//头数据长度，网络字节序
	unsigned int		payload_len;			//原始数据长度，网络字节序
	unsigned int		image_width;			//raw图的宽，网络字节序
	unsigned int		image_hight;			//raw图的高，网络字节序
	unsigned int		image_stride;			//raw图像跨度（字节为单位, imageStride/imageWidth=每个像素占的字节数，目前raw图像是高14bit有效），网络字节序
	unsigned int		mirror_mode;			//raw图像镜像模式（: 正常，：水平，：垂直，：水平垂直）,，网络字节序
	unsigned int		capture_timeh;          //raw图像时间戳高位，网络字节序
	unsigned int		capture_timel;          //raw图像时间戳低位，网络字节序
	unsigned int		sequence_id;			//序号ID,，网络字节序
	char				reserve[64];			//预留字段
}ST_ThermalAVData;


// 硬件能力
typedef struct _dev_hw_cap_t_
{
	unsigned short				chn_num;	              //通道数（摄像机数）
	unsigned char				audio_in_num;	          //音频输入个数
	unsigned char				sound_type;	              //音频通道类型
	unsigned char				audio_out_num;	          //音频输出个数
	unsigned char				alarm_in_num;	          //报警输入个数
	unsigned char				alarm_out_num;	          //报警输出个数
	unsigned char				rs485_num;	              //RS485串口个数
	unsigned char				rs232_Num;	              //RS232串口个数
	unsigned char				netcard_num;	          //有线网卡个数
	unsigned char				usb_num;	              //USB个数
	unsigned char				sd_num;	                  //SD卡的个数
	unsigned char				hd_num;	                  //硬盘的个数
	unsigned char				is_wifi;	              //是否支持wifi
	unsigned char				is_poe;	                  //是否支持POE
	unsigned char				is_ir;	                  //是否支持红外
	unsigned char				is_pir;	                  //是否支持PIR
	unsigned char				is_bnc;	                  //是否支持模拟通道
	unsigned char				is_ptz;	                  //是否支持内置云台
	unsigned char				is_face;	                //是否支持人脸库
	unsigned short              virtualPTZ_Type;            //虚拟PTZ类型
	unsigned short              aiDetec_Type;             //AI 检测抓怕类型，人脸、车牌、人形，人脸识别，改类型丰富，解决各类搭配问题，在hardware可配置
	unsigned char				is_faceDetect;	           //是否支持人脸检测
	unsigned char               resv;                      // 保留字段
}dev_hw_cap_t;

// 软件能力
typedef struct _dev_sw_cap_t_
{
	unsigned char				max_user_num;	    //最大登录用户数
	unsigned char				max_preview_num;	//最大实时预览路数
	unsigned char				max_pb_num;	        //最大回放和下载路数
	unsigned char               resv;               //保留字段
}dev_sw_cap_t;

// 音频能力
typedef struct _dev_audio_cap_t_
{
	unsigned char               is_interphone;       //对讲
	unsigned char               is_audio_in;         //音频输入
	unsigned char               is_audio_out;        //音频输出
	unsigned char               resv;                //保留字段
}dev_audio_cap_t;

// 设备概要信息
typedef struct _dev_general_info_t_
{
	char	 dev_id[MAX_DEV_NAME_LEN];	                  //设备Id
	char	 dev_name[MAX_DEV_NAME_LEN];				  //设备名称
	char	 dev_ip[MAX_IP_BUF_LEN];					  //设备IP
	char	 dev_mac[MAX_IP_BUF_LEN];					  //MAC地址
	char	 dev_man_name[MAX_DEV_NAME_LEN];              //厂商名称
	char	 dev_man_id[MAX_DEV_NAME_LEN];                //设备型号
	char	 prod_model[MAX_DEV_NAME_LEN];	              //产品模组
	char     dev_sn[MAX_DEV_NAME_LEN];	                  //SN
	char     sw_info[MAX_DEV_NAME_LEN];	                  //软件包信息
	char     hw_info[MAX_DEV_NAME_LEN];	                  //硬件信息
	short    dev_type;                                   //设备类型
	unsigned short dev_port;                          //设备端口
}dev_general_info_t;

// 设备名称
typedef struct _dev_name_t_
{
	char dev_name[MAX_DEV_NAME_LEN];				  //设备名称
}dev_name_t;

// 设备时间
typedef struct _dev_time_t_
{
	unsigned short         year;                     //年
	unsigned char          mon;                      //月
	unsigned char          day;                      //日
	unsigned char          hour;                     //时
	unsigned char          min;                      //分
	unsigned char          sec;                      //秒
	unsigned char          resv;                     //保留
}dev_time_t;

// ntp参数
typedef struct _ntp_param_t_
{
	char             serv_ip[MAX_IP_BUF_LEN];              //server ip
	unsigned short   serv_port;                            //server port
	unsigned char    is_enable;                            //是否启用ntp
	unsigned char    ip_proto_ver;                         //ip 协议版本
	unsigned int     ntp_time;                             //ntp time
}ntp_param_t;

typedef struct _dev_port_t_
{
	char                        dev_id[MAX_DEV_NAME_LEN];             //设备id
	unsigned short				ctrl_port;	                          //网络视频设备的设备网络控制端口
	unsigned short				av_port;	                          //网络视频设备的TCP音视频端口
	unsigned short				http_port;	                          //网络视频设备的设备HTTP端口
	unsigned short				rtsp_port;	                          //网络视频设备的设备RTSP端口
}dev_port_t;

// PTZ 操作
typedef enum ptz_operation_e
{
	PTZ_STOP = 0,  //停止	
	PTZ_UP = 1,        //向上
	PTZ_DOWN = 2,      //向下	
	PTZ_LEFT = 3,      //左	
	PTZ_RIGHT = 4,     //右		
	PTZ_LEFT_UP = 5,   //左上	
	PTZ_LEFT_DOWN = 6, //左下
	PTZ_RIGHT_UP = 7,  //右上
	PTZ_RIGHT_DOWN = 8, //右下
	PTZ_ZOOM_IN = 9,     //拉近
	PTZ_ZOOM_OUT = 10,   //拉远	
	PTZ_FOCUS_FAR = 11,  //远焦
	PTZ_FOCUS_NEAR = 12,  //近焦	
	PTZ_IRIS_INC = 13,   //光圈变大
	PTZ_IRIS_DEC = 14,   //光圈减小
	PTZ_PRESET_SET = 15, //预置位设置
	PTZ_PRESET_CALL = 16, //预置位调用
	PTZ_PRESET_DEL = 17,  //预置位删除
	//PTZ_TRACE_SET= 18,   //轨迹设置
	//PTZ_TRACE_CALL= 19,  //轨迹调用
	//PTZ_TRACE_DEL= 20,   //轨迹删除	
	PTZ_SCAN_CALL=21,    //扫描调用
	PTZ_SCAN_SET_START = 22,  //设置扫描起始点
	PTZ_SCAN_SET_STOP = 23,  //设置扫描结束点
	PTZ_AUTO_FOCUS = 24,     //自动聚焦	
	PTZ_AUTO_IRIS = 25,     //自动光圈	
	PTZ_START_AUTO_STUDY = 26, //开始自学习
	PTZ_END_AUTO_STUDY = 27,  //结束自学习
	PTZ_RUN_AUTO_STUDY = 28,  //自学习调用
	PTZ_RESET = 29,           //复位
	PTZ_3D_ORIENTATION = 30,  //三维智能定位
	PTZ_TOUR_SET_START = 31,   //设置巡游起始点
	PTZ_TOUR_ADD_PRESET = 32,  //添加巡游预置点
	PTZ_TOUR_SET_END = 33,    //设置巡游结束点
	PTZ_TOUR_RUN = 34,        //调用巡游
	PTZ_TOUR_PAUSE = 35,       //暂停巡游
	PTZ_TOUR_DEL = 36,        //删除巡游
	PTZ_TOUR_CONTINUE = 200,   //继续巡游（和暂停巡游配对使用）
	PTZ_KEEPER_SET = 37,      //看守位设置
	PTZ_KEEPER_RUN = 38,      //运行看守位
	PTZ_RUN_BRUSH = 39,      //雨刷运行
	PTZ_OPEN_LIGHT = 40,       //打开灯
	PTZ_CLOSE_LIGHT = 41,      //关闭灯
	PTZ_SCAN_REMOVE = 44,      //删除扫描
	PTZ_REMOVE_AUTO_STUDY = 45,     //删除自学习
	PTZ_INFRARED_CTRL = 46,        //红外灯控制
	PTZ_GET_PTZ_POSTION_REQ = 47,   //请求获取PTZ位置
	PTZ_GET_PTZ_POSTION_RESP = 48,   //PTZ位置应答
	PTZ_SET_PTZ_POSTION = 49,        //设置PTZ位置
	PTZ_SET_PTZ_NORTH_POSTION = 50,   //设置正北位置
	PTZ_GET_PRESET_REQ = 51,      //获取预置位请求
	PTZ_GET_PRESET_RESP = 52,     //获取预置位应答
	PTZ_GET_TOUR_REQ = 53,    //获取巡游请求
	PTZ_GET_TOUR_RESP = 54,    //获取巡游应答
	PTZ_GET_SCAN_REQ = 55,    //获取扫描请求
	PTZ_GET_SCAN_RESP = 56,    //获取扫描应答
	PTZ_GET_AUTO_STUDY_REQ = 57,    //获取自学习请求
	PTZ_GET_AUTO_STUDY_RESP = 58,    //获取自学习应答
	PTZ_GET_KEEPER_REQ = 59,     //获取看守卫请求
	PTZ_GET_KEEPER_RESP = 60,    //获取看守卫应答
	PTZ_INFRARED_STRL_V2 = 61,   //红外灯控制扩展命令
	PTZ_INFRARED_STRL_V2_REQ = 62,   //请求红外灯控制参数命令
	PTZ_STOP_BRUSH = 63,       //雨刷停止
	PTZ_360_ROTATE_SCAN = 64,  //360°旋转扫描
	PTZ_PERPENDICVULAR_SCAN = 65,  //垂直扫描
	PTZ_HEART_BEAT = 66,       //心跳
	PTZ_INFRARED_CTRL_V2_RESP = 67, //请求红外灯控制参数命令应答
	//PTZ_ZOOM_SPEED= 67,            //镜头拉近拉远速度值设置，设置为与华为兼容，该值景阳暂时不处理，不造成冲突
	PTZ_GET_ALARM_IO_START_REQ = 70,   //请求获取报警IO状态
	PTZ_GET_ALARM_IO_START_RESP = 71,  //报警IO状态应答
	PTZ_PT_STOP_STATUS_RESP = 72,     //PT停止状态查询
	PTZ_PT_POS_AUTO_RESP = 73,        //自动上报PT坐标
	PTA_ALARM_IO_STATUS_AUTO_RESP = 74,   //自动上报IO报警状态
	PTZ_GET_ZOOM_VALUE = 75,       //镜头变倍值
	PTZ_GET_PTZ_VERSION = 76,      //获取PTZ版本号
	PTZ_GET_MCU_TEMPERATURE = 77,   //获取MCU温度
	PTZ_LOAD_DEFAULT = 78,          //清理所有操作	
	PTZ_GET_PT_POSTION = 79,
	PTZ_SET_VERTICAL_MAX_POSTION = 80,
	PTZ_LENS_RESET = 81,	/*自动聚焦镜头(包括ABF)复位*//*BOOL*/
	PTZ_AUTO_TRACK = 82,
	PTZ_GET_PTZ_ACTION_STATUS_REQ = 83,	//请求获取PTZ运动状态
	PTZ_GET_PTZ_ACTION_STATUS_RESP = 84,	//PTZ运动状态应答
	PTZ_SET_WIPER_MODE = 85,	//设置雨刷模式
	PTZ_GET_WIPER_MODE = 86,	//获取雨刷模式
	PTZ_SET_PTZ_POWER_SAVE = 87,	//设置PTZ省电
	PTZ_GET_PTZ_POWER_SAVE = 88,	//获取PTZ省电
	PTZ_SET_PT_LIMIT_POS = 89,	//设置PT限制位置
	PTZ_GET_PT_LIMIT_POS_REQ = 90,	//获取PT限制位置请求
	PTZ_GET_PT_LIMIT_POS_RESP = 91,	//获取PT限制位置应答
	PTZ_CLEAR_PT_LIMIT_POS = 92,	//清除PT限制位置
	PTZ_SET_PT_SELFCHECK = 93,	//设置PT自检
	PTZ_GET_PT_SELFCHECK = 94,	//获取PT自检
	PTZ_SET_ORIENTATION = 95,	//设置安装方式
	PTZ_GET_ORIENTATION = 96,	//获取安装方式
	PTZ_SET_SHORTCUT = 97,	//设置快捷方式
	PTZ_GET_SHORTCUT = 98,	//获取快捷方式
	PTZ_SET_DN_MODE = 99,	//设置日夜模式
	PTZ_SET_WHITE_LIGHT = 100,	//设置白光灯状态
	PTZ_GET_WHITE_LIGHT = 101,	//获取白光灯状态
	PTZ_GET_DN_MODE = 102,	//获取日夜模式
	PTZ_SET_ZOOM_VALUE = 103,	//设置变倍值
	PTZ_SET_FOCUS_VALUE = 104,	//设置聚焦值
	PTZ_GET_FOCUS_VALUE = 105,	//获取聚焦值
	PTZ_BOW_SCAN = 110,	//弓形扫描	
	PTZ_BOW_SCAN_SET_STARTPOINT = 111,	//设置弓形扫描起始点	
	PTZ_BOW_SCAN_SET_STOPPOINT = 112,	//设置弓形扫描结束点	
	PTZ_BOW_SCAN_REMOVE = 113,	//删除弓形扫描
	PTZ_BOW_SCAN_PAUSE = 114,	//暂停弓形扫描
	PTZ_BOW_SCAN_CONTINUE = 115,	//继续弓形扫描
	PTZ_OPEN_DEFOG = 120,	//打开透雾
	PTZ_CLOSE_DEFOG = 121	//关闭透雾   	
}ptz_operation_e;
enum SET_PTZ_POSION_TYPE
{
	POSTION_TYPE_PAN = 0x01, //水平位置 
	POSTION_TYPE_TILE = 0x02, //垂直位置
	POSTION_TYPE_ZOOM = 0x04  //放大倍数
};
enum PTZ_DIRECTION
{
	EAST = 0x00,
	SOUTHEAST = 0x01,
	SOUTH = 0x02,
	SOUTHWEST = 0x03,
	WEST = 0x04,
	NORTHWEST = 0x05,
	NORTH = 0x06,
	NORTHEAST = 0x07
};
enum PTZ_RUN_KEEPER
{
	RUN_KEEPER_OFF = 0x00,	//关闭看守位
	RUN_KEEPER_ON = 0x02  //开启看守位
};
enum PTZ_ZOOM
{
	ZOOM_SPEED_MIN = 0x00, //镜头拉近拉远速度值的最小值
	ZOOM_SPEED_MAX = 0x3F  //镜头拉近拉远速度值的最大值
};
enum PTZ_ROTATE_TYPE
{
	ROTATE_TYPE_GEAR = 0x00,	//挡位（1-64）
	ROTATE_TYPE_SPEED = 0x01,	//速度
	ROTATE_TYPE_DEGREE = 0x02	//度数
};
enum
{
	CONFIGURE_PRESET = 0,
	CONFIGURE_SCAN = 1,
	CONFIGURE_TRACK = 2,
	CONFIGURE_TOUR = 3,
	CONFIGURE_KEEPER = 4,
	CONFIGURE_GET_SPEED = 6,
};

//时间参数
typedef struct  _dev_time_zone_param_t_
{
	int				nTimeZone;												//时区

	unsigned char	bDSTOpenFlag;											//夏令时开启标志

	int				nBeginMonth;											//夏令时开始月份
	int				nBeginWeekly;											//夏令时开始周（一月中的第几周）
	int				nBeginWeekDays;											//星期几
	unsigned int	nBeginTime;												//开始时间

	int				nEndMonth;												//夏令时结束月份
	int				nEndWeekly;												//夏令时结束周（一月中的第几周）
	int				nEndWeekDays;											//星期几
	unsigned int	nEndTime;												//结束时间

}dev_time_zone_param_t;

typedef struct _dev_modify_password_info_t_
{
	char dev_old_password[64];
	char dev_new_password[64];
}dev_modify_password_info_t;

typedef struct _dev_user_info_t_
{
	char dev_username[64];
	char dev_password[64];
}dev_user_info_t;




//计划时间
typedef struct _schedule_time_
{
	int week_day;
	unsigned long start_time;
	unsigned long end_time;
}schedule_time;

typedef struct _schedule_time_list_
{
	int					schedule_time_count;
	schedule_time	    time_list[CONST_MAX_PERIOD_RECORD_TIME_NUM];
}schedule_time_list;

typedef struct _detection_area_
{
	int width_num;
	int high_num;
	char data[512];
}detection_area;

typedef struct _time_struct_
{
	int				time_zone;				//时区
	unsigned short	day_light_saving_time;	//夏令营时
	unsigned short	year;					//年
	unsigned short 	month;					//月[1,12]
	unsigned short 	day;					//日[1,31]
	unsigned short 	day_of_week;				//星期几[0,6]
	unsigned short 	hour;					//时[0,23]
	unsigned short 	minute;				//分[0,59]
	unsigned short 	second;				//秒[0,59]
	int 			milli_seconds;			//微妙[0,1000000]
}time_struct;

//////////////////motion detection///////////////////////////////
typedef struct _alarm_source_param_
{
	int					 enable;
	int					 alarm_interval;
	int				     check_block_num;
	int					 sensitivity;
	int                  time_list_size;
	int					 area_datalen;
	detection_area		 area;
	schedule_time		 time_list[200];
}alarm_source_param;

//报警动作
typedef struct _alarm_action_
{
	int action_type;	//报警源类型
	int action_id;		//报警源ID
	char action_name[MAX_DEV_NAME_LEN];		//报警源名称
}alarm_action;

//PTZ报警参数
typedef struct _ptz_action_param_
{
	int ptz_action_type;	//操作类型（预置位、轨迹等）
	int ptz_action_id;		//操作ID（用户之前设置的预置位ID、轨迹ID等）
	int ptz_channel_id;		//PTZ通道ID
	alarm_action  alarm_act;
}ptz_action_param;

//报警输出参数
typedef struct _alarm_out_param_
{
	char dev_id[MAX_LENGTH_DEVICEID + 1];		//设备id
	int alarm_out_id;	//报警输出端口的ID号
	int alarm_out_flag;	//报警输出标志
	int event_type_id;	//报警事件类型
	int alarm_time;		//报警输出时间
	alarm_action  alarm_act;
}alarm_out_param;

//联动录像参数
typedef struct _record_act_param_
{
	unsigned char pre_record_flag; //是否开启预录
	int			delay_record_time;	//延录制时长
	alarm_action	alarm_act;
}record_act_param;


//联动报警参数
typedef struct _alarm_link_t_
{
	int			action_type;
	int			action_id;
}alarm_link;

typedef struct _mot_detect_param_
{
	alarm_source_param  objSourceParam;
	alarm_link          objLinkParamList[10];
	ptz_action_param    objPtzParamList[10];
	alarm_out_param		objAlarmOutList[10];
	record_act_param	objRecordActionList[10];
}mot_detect_param;

//IO报警内置类型参数
typedef struct _io_alarm_insource_para_
{
	alarm_action		alarm_act;
	unsigned char		enable_flag;	//开启标记
	int					alarm_inval;	//报警间隔
	int					valid_level;	//有效电平
	schedule_time_list  schedule_para;
}io_alarm_insource_para;

typedef struct _io_alarm_event_para_
{
	io_alarm_insource_para   insource_para;
	int					linkage_param_count;
	alarm_link          link_param_list[CONST_MAX_ALARM_IO_NUM];
	int					ptz_action_action_param_list_count;
	ptz_action_param    ptz_param_list[CONST_MAX_ALARM_IO_NUM];
	int					alarm_out_count;
	alarm_out_param		alarm_out_list[CONST_MAX_ALARM_IO_NUM];
	int					record_action_param_list_count;
	record_act_param	record_action_list[CONST_MAX_ALARM_IO_NUM];
}io_alarm_event_para;

typedef struct _io_alarm_event_para_list_
{
	int		alarm_event_list_count;
	io_alarm_event_para alarm_event_list[CONST_MAX_ALARM_IO_NUM];
}io_alarm_event_para_list;

typedef struct _disk_alarm_source_para_
{
	alarm_action		alarm_act;
	unsigned short		enable_flag;	//是否启动磁盘报警(false：不启动， true：启动)
	int		            alarm_inval;	//上报间隔，单位为秒，最小间隔为10秒，最大为86400秒(1天)
	int					alarm_thresold;//报警阈值, 单位为百分比
	unsigned short		disk_full_enable_flag;
	unsigned short		disk_error_enable_flag;
	unsigned short		no_disk_enable_flag;
	schedule_time_list  schedule_para;
}disk_alarm_source_para;

//磁盘报警参数
typedef struct _disk_alarm_event_para_
{
	disk_alarm_source_para disk_alarm_source;
	int					linkage_param_count;
	alarm_link          link_param_list[CONST_MAX_ALARM_OUT_NUM];
	int					ptz_action_action_param_list_count;
	ptz_action_param    ptz_param_list[CONST_MAX_ALARM_OUT_NUM];
	int					alarm_out_count;
	alarm_out_param		alarm_out_list[CONST_MAX_ALARM_OUT_NUM];
	int					record_action_param_list_count;
	record_act_param	record_action_list[CONST_MAX_ALARM_OUT_NUM];
}disk_alarm_event_para;

typedef struct _disk_alarm_event_para_list_
{
	int					disk_alarm_event_count;
	disk_alarm_event_para disk_alarm_event_list[CONST_MAX_ALARM_OUT_NUM];
}disk_alarm_event_para_list;



//查询条件信息
typedef struct _qry_info_
{
	char				dev_id[MAX_LENGTH_DEVICEID + 1];		//设备ID
	long				channel_id;								//通道号
	long				record_mode;							//查询模式(录像查询or快照查询)
	long				select_mode;							//查询模式(0:所有；1：按类型查询；2：按时间查询)
	long				major_type;								//主类型
	long				minor_type;								//次类型
	long				precision;								//精度
	int					record_segment_interval;		////查询段时间长度（每段最长时间跨度）
	time_struct			begin_time;						//开始时间
	time_struct			end_time;						//结束时间

}qry_info;


typedef struct _qry_info_list_
{
	int				qry_info_count;
	qry_info		qry_info_list[CONST_MAX_ALARM_OUT_NUM];
}qry_info_para_list;

typedef struct  _alarm_info_qry_
{
	char			dev_ip[MAX_IP_BUF_LEN];					//设备IP
	char			dev_id[MAX_LENGTH_DEVICEID + 1];
	int				source_id;	//报警源Id
	int				select_mode;	//查询模式 :SELECT_MODE_ALL
	char			source_name[MAX_DEV_NAME_LEN];	//源名称
	int				major_type;	//报警主类型
	int				minor_type;	//报警子类型
	unsigned long				alarm_begin_time;	//查询开始时间
	time_struct				alarm_begin_time_struct;	//
	unsigned long				alarm_end_time;	//查询结束时间
	time_struct				alarm_end_time_struct;	//
}alarm_info_qry;


#ifdef WIN32
typedef enum sdks_zoomin_graduate_e
{
	SDKS_ZOOMIN_GRADUATE_MIN = 0, ///<放大倍率
	SDKS_ZOOMIN_GRADUATE_1,       ///<放大倍率加0.25 
	SDKS_ZOOMIN_GRADUATE_2,       ///<放大倍率加0.5 
	SDKS_ZOOMIN_GRADUATE_3,       ///<放大倍率加0.75 
	SDKS_ZOOMIN_GRADUATE_4,       ///<放大倍率加1.0 
	SDKS_ZOOMIN_GRADUATE_5,       ///<放大倍率加1.25 
	SDKS_ZOOMIN_GRADUATE_6,       ///<放大倍率加1.5 
	SDKS_ZOOMIN_GRADUATE_7,       ///<放大倍率加1.75 
	SDKS_ZOOMIN_GRADUATE_8,       ///<放大倍率加2.0 
	SDKS_ZOOMIN_GRADUATE_9,       ///<放大倍率加2.25 
	SDKS_ZOOMIN_GRADUATE_10,      ///<放大倍率加2.5 
	SDKS_ZOOMIN_GRADUATE_11,      ///<放大倍率加2.75 
	SDKS_ZOOMIN_GRADUATE_12,      ///<放大倍率加3.0
	SDKS_ZOOMIN_GRADUATE_13,      ///<放大倍率加3.25 
	SDKS_ZOOMIN_GRADUATE_14,      ///<放大倍率加3.5 
	SDKS_ZOOMIN_GRADUATE_15,      ///<放大倍率加3.75 
	SDKS_ZOOMIN_GRADUATE_16,      ///<放大倍率加4.0 
	SDKS_ZOOMIN_GRADUATE_17,      ///<放大倍率加4.25 
	SDKS_ZOOMIN_GRADUATE_18,      ///<放大倍率加4.5 
	SDKS_ZOOMIN_GRADUATE_19,      ///<放大倍率加4.75
	SDKS_ZOOMIN_GRADUATE_20,	  ///<放大倍率加5.0 
	SDKS_ZOOMIN_GRADUATE_21,	  ///<放大倍率加5.25
	SDKS_ZOOMIN_GRADUATE_22,	  ///<放大倍率加5.5 
	SDKS_ZOOMIN_GRADUATE_23,	  ///<放大倍率加5.75
	SDKS_ZOOMIN_GRADUATE_24,	  ///<放大倍率加6.0 
	SDKS_ZOOMIN_GRADUATE_25,	  ///<放大倍率加6.25
	SDKS_ZOOMIN_GRADUATE_26,	  ///<放大倍率加6.5 
	SDKS_ZOOMIN_GRADUATE_27,	  ///<放大倍率加6.75
	SDKS_ZOOMIN_GRADUATE_28,	  ///<放大倍率加7.0 
	SDKS_ZOOMIN_GRADUATE_29,	  ///<放大倍率加7.25
	SDKS_ZOOMIN_GRADUATE_30,	  ///<放大倍率加7.5 
	SDKS_ZOOMIN_GRADUATE_31,	  ///<放大倍率加7.75
	SDKS_ZOOMIN_GRADUATE_32,	  ///<放大倍率加8.0 
	SDKS_ZOOMIN_GRADUATE_33,	  ///<放大倍率加8.25
	SDKS_ZOOMIN_GRADUATE_34,	  ///<放大倍率加8.5 
	SDKS_ZOOMIN_GRADUATE_35,	  ///<放大倍率加8.75
	SDKS_ZOOMIN_GRADUATE_36,	  ///<放大倍率加9.0 
	SDKS_ZOOMIN_GRADUATE_37,	  ///<放大倍率加9.25
	SDKS_ZOOMIN_GRADUATE_38,	  ///<放大倍率加9.5 
	SDKS_ZOOMIN_GRADUATE_39,	  ///<放大倍率加9.75
	SDKS_ZOOMIN_GRADUATE_40,	  ///<放大倍率加10.0 
	SDKS_ZOOMIN_GRADUATE_41,	  ///<放大倍率加10.25
	SDKS_ZOOMIN_GRADUATE_42,	  ///<放大倍率加10.5 
	SDKS_ZOOMIN_GRADUATE_43,	  ///<放大倍率加10.75
	SDKS_ZOOMIN_GRADUATE_44,	  ///<放大倍率加11.0 
	SDKS_ZOOMIN_GRADUATE_45,	  ///<放大倍率加11.25
	SDKS_ZOOMIN_GRADUATE_46,	  ///<放大倍率加11.5 
	SDKS_ZOOMIN_GRADUATE_47,	  ///<放大倍率加11.75
	SDKS_ZOOMIN_GRADUATE_48,	  ///<放大倍率加12.0 
	SDKS_ZOOMIN_GRADUATE_49,	  ///<放大倍率加12.25
	SDKS_ZOOMIN_GRADUATE_50,	  ///<放大倍率加12.5 
	SDKS_ZOOMIN_GRADUATE_51,	  ///<放大倍率加12.75
	SDKS_ZOOMIN_GRADUATE_52,	  ///<放大倍率加13.0 
	SDKS_ZOOMIN_GRADUATE_53,	  ///<放大倍率加13.25
	SDKS_ZOOMIN_GRADUATE_54,	  ///<放大倍率加13.5 
	SDKS_ZOOMIN_GRADUATE_55,	  ///<放大倍率加13.75
	SDKS_ZOOMIN_GRADUATE_56,	  ///<放大倍率加14.0 
	SDKS_ZOOMIN_GRADUATE_57,	  ///<放大倍率加14.25
	SDKS_ZOOMIN_GRADUATE_58,	  ///<放大倍率加14.5 
	SDKS_ZOOMIN_GRADUATE_59,	  ///<放大倍率加14.75
	SDKS_ZOOMIN_GRADUATE_60,	  ///<放大倍率加15.0 
	SDKS_ZOOMIN_GRADUATE_MAX
}sdks_zoomin_graduate_e;

#endif

#endif
