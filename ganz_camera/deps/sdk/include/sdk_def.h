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
typedef void(*SDK_FACE_CB)(unsigned int handle, int pic_type,void* p_data, int *data_len, void** p_result, void* p_obj);//pic_type: 1�׿⣬ 2ʵʱ��
typedef void(*SDK_MICROPHONE_CB)(unsigned int handle, void* p_data, int *data_len, void* p_obj);
typedef void(*SDK_INTERCOM_DB_CB)(unsigned int db, void*p_obj);

#define MAX_DEV_NAME_LEN  128
#define MAX_IP_BUF_LEN  128
#define CONST_MAX_ALARM_IO_NUM  10					//IO�������֧�ָ���
#define CONST_MAX_PERIOD_RECORD_TIME_NUM  16				//��������¼���ʱ��θ���
#define CONST_MAX_ALARM_OUT_NUM   16						//�������󱨾����ͨ������	
#define MAX_LENGTH_DEVICEID   32		//�豸id����
typedef enum  video_stream_type_e
{
	STREAM_TYPE_1 = 1,                     // HD
	STREAM_TYPE_2,                         // Smooth
	STREAM_TYPE_3
}video_stream_type_e;

enum Device_Type     //�豸����
{
	IPCAMERA = 1,	//����������豸
	DVR = 2,	//������Ƶ¼����豸
	DVS = 3,	//������Ƶ�������豸
	IPDOME = 4,	//���������
	NVR = 5,	//NVR
	ONVIF_DEVICE = 6,	//Onvif �豸
	DECODER = 7,	//������
	LPR = 8,	//����ʶ�������
	FISHEYE = 9,    //�����豸���͡�
	NVR_2 = 10,   //��ʾ4.0��NVR
	IPP = 11,   //��Ŀȫ�������
	THERMAL_DEVICE = 13,	//�ȳ����豸
	HUMAN_TEMPERATURE = 14,//���������
	FACE_DETECT = 15,//�������
	VEHICLE_DETECT = 16,//���ڳ���-��о
	THERMAL_DOUBLE_SENSOR_DEVICE = 17,	//�ȳ���˫����ͨ�����豸
	AIMULTI_OBJECT_DETECT = 18,//AI MULTI OBJECT DETECT
	THERMAL_DOUBLE_SENSOR_DOME_DEVICE = 19,	//�ȳ���˫����ͨ�����豸(�ɼ���ӻ�о)
	DOMECORE_THERMAL_DEVICE = 20,			//�а���̨˫ip��pmd1030��о
	FACE_DEVICE = 21,  //�������ܺ���
	HK_DVR = 100,	//HK DVR
	RS_DVR = 101,	//����DVR
	DH_DVR = 102,	//��DVR
	VIRTUAL_NVR = 103,   //���ڼ���NVR�ͻ�����ʹ�õı���Server
	DOMECORE = 104	//��о
};

typedef struct jy_facebase_info    //ʵʱ��
{
	int  key_id;		//4�ֽ�, ����
	int  channel;    //4�ֽ�, ͨ��id
	int  similarity;	//4�ֽ�, ���ƶ�
	long long time;		//8�ֽ�, ץ��ʱ��㣬��λus
	char name[36];		//���32�ֽ�,��������,4λ�ǽ�����0
	char id[36];		//���32�ֽ�,��������
	char group[36];		//���32�ֽ�,����������
	char type[36];		//���32�ֽ�,��������
	long long date;     //8�ֽ�, �������գ���λus
	unsigned short gender;      //2�ֽ�,�����Ա�		
}jy_facebase_info;

typedef struct jy_face_info       //�׿�
{
	int  key_id;				//����
	int  expired;				//�Ƿ����ڣ�0�Ӳ����ڣ�1�Ѿ����ڣ�2δ���ڣ�
	char name[36];				//��������
	char identity[36];			//������ݱ�ʶid�����֤���߹��ţ�
	char group[36];				//����������
	char type[36];				//�������ͣ�ͨ�����˵�ְλ��Ϣ��������ʦ��ѧ����	
	long long   birthday;		//������������
	long long	s_time;			//��ʼʱ�䣬��λΪ΢��
	long long	e_time;			//����ʱ�䣬��λΪ΢��
	unsigned short  gender;		//�����Ա�
	unsigned short  similarity; //��׿�����ƶ�
	float  temperature;			//�����¶�ֵ
	//char padding[64];	//Ԥ���ռ䣬64�ֽ�
}jy_face_info;

typedef struct tagFacePicData
{
	int                 pic_flag;					//1(�׿�)��2��ʵʱ�⣩
	char*				pszPicDate;					//����(�׿�)
	unsigned long		nPicDataLength;				//������Ч����(�׿�)
	jy_face_info        param;						//�׿�
	//char*				pszData;					//����(ʵʱ��)
	//unsigned long		nDataLength;				//������Ч����(ʵʱ��)
	jy_facebase_info    base_param;                //ʵʱ��
}tagFacePicData;

typedef struct FRAME_INFO {
	long nWidth;	// �������λΪ���أ��������Ƶ������Ϊ0
	long nHeight;	// ����ߣ���λΪ���أ��������Ƶ������Ϊ0
	long nStamp;	// ʱ����Ϣ����λ����
	long nType;		//�������ͣ����±�
	long nFrameRate;// ����ʱ������ͼ��֡��
}FRAME_INFO;

typedef struct tagAVFrameData
{
	long					nStreamFormat;						//1��ʾԭʼ����2��ʾTS�������3��ʾԭʼ��������4��ʾPS�������
	long					nESStreamType;						//ԭʼ�����ͣ�1��ʾ��Ƶ��2��ʾ��Ƶ��
	long					nEncoderType;						//�����ʽ��
	long					nCameraNo;							//������ţ���ʾ�������Եڼ�·
	unsigned long			nSequenceId;						//����֡���
	long					nFrameType;							//����֡����,1��ʾI֡, 2��ʾP֡, 0��ʾδ֪����
	long long				nAbsoluteTimeStamp;					//���ݲɼ�ʱ�������λΪ΢��
	long long				nRelativeTimeStamp;					//���ݲɼ�ʱ�������λΪ΢��
	char*					pszData;							//����
	unsigned long			nDataLength;						//������Ч����
	long					nFrameRate;							//֡��
	long					nBitRate;							//��ǰ���ʡ�
	long					nImageFormatId;						//��ǰ��ʽ
	long					nImageWidth;						//��Ƶ���
	long					nImageHeight;						//��Ƶ�߶�
	long					nVideoSystem;						//��ǰ��Ƶ��ʽ
	unsigned long			nFrameBufLen;						//��ǰ���峤��

	long					nStreamId;							// ��ID
	long					nTimezone;							// ʱ��
	long					nDaylightSavingTime;				//����ʱ
}ST_AVFrameData;

typedef struct tagTargeHead
{
	int					targe_id;				//Ŀ��Id
	unsigned short		type;					//���ͣ����������壬���ƣ�������
	unsigned short		x;						//��������
	unsigned short		y;						//���������
	unsigned short		w;						//���ο��
	unsigned short		h;						//���θ߶�
	unsigned short		attr_data_len;			//�������ݳ���

}tagTargeHead;

typedef struct tagPersonFace
{
	//tagTargeHead		headInfo;
	unsigned short		confidence;				//���Ŷ�
	unsigned short		yaw;					//���
	unsigned short		tilt;					//б��
	unsigned short		pitch;					//����
	unsigned short		gender;					//�Ա�
	unsigned short		age;					//����
	unsigned short		complexion;				//��ɫ
	unsigned short		minority;				//����
	float				temperature;			//�����¶�
	char				reserve[32];			//Ԥ���ֶ�
	unsigned int		land_mark_size;			//���������
	//char*		        land_mark_data;			//����������
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
	unsigned short		have_plate;//�Ƿ��г���,0��,1��
	unsigned short		plate_angleH;//! �Ƕ�
	unsigned short		plate_angleV;//! �Ƕ�
	unsigned short		plate_color;	//! <������ɫ
	unsigned short		plate_type;	//! <��������
	unsigned short		plate_confidence; //! <�������Ŷ�
	unsigned short		plate_country;//! < �Ĺ�����
	unsigned short		char_num;		//! <�����ַ�����
	char plate_num[32];		//! <���ƺ�
	char plate_char_confidence[32];//! <���ƺ�ÿ���ַ������Ŷ�
}plate_info_s;

typedef struct _vehicle_info_s
{
	//tagTargeHead		headInfo;
	unsigned short		have_plate;//�Ƿ��г���,0��,1��
	unsigned short		vehicle_plate_angleH;	//!<  ����ˮƽ��б�Ƕ�;
	unsigned short		vehicle_plate_angleV;	//!<  ���ƴ�ֱ��б�Ƕ�;
	unsigned short		vehicle_color;	//!<  ������ɫ;
	unsigned short		vehicle_type; //!<  �������ͣ����������ǻ�����;
	unsigned short		confidence;
	unsigned short		vehicle_model; //!<  �����ͺţ�����������С������
	unsigned short		vehicle_speed;	//!<  ����ʹ���ٶ�;
	unsigned short		Vehicle_moving;	//!< �����Ƿ����ƶ�
	unsigned short		char_num;		//! <�����ַ�����
	char				plate_num[32];		//! <���ƺ�
	char				plate_char_confidence[32];//! <���ƺ�ÿ���ַ������Ŷ�
	unsigned short		Move_direction;//!< �����ƶ�����
	unsigned short		Move_direction_confidence;//!< �����ƶ�������Ŷ�
	unsigned short		Vehicle_facing;//!< ��������
	unsigned short		Vehicle_facing_confidence;//!< �����������Ŷ�
	char				trademark_utf8[32];//!< ����Ʒ��
	unsigned short		trademark_utf8_confidence;//!< ����Ʒ�����Ŷ�
	unsigned short		Reserve;
}vehicle_info_s;

typedef struct tagDetectFaceData
{
	char				magic[4];				//ħ���ţ�Ĭ��0xffff ffff
	int					vesion;					//����Э��汾�ţ������ֽ���
	unsigned int		total_len;				//�����ṹ���ȣ������ֽ���
	unsigned int		picture_len;			//ͼƬ���ݳ��ȣ������ֽ���
	unsigned short		full_image_width;		//��ͼ�Ŀ�Full_corp=0��Ч�������ֽ���
	unsigned short		full_image_hight;		//��ͼ�ĸߣ�Full_corp=0��Ч�������ֽ���												
	//long long			capture_time;           //���Ŀ��ʱ�̵�λ�룬�����ֽ���
	unsigned int capture_timeh;
	unsigned int capture_timel;
	unsigned int		sequence_id;			//���ID,�������ֽ���
	char				full_crop;				//��ͼСͼ0����1��С��
	char				reserve[31];			//Ԥ���ֶ�
	unsigned int		target_size;			//���Ŀ������������ֽ���
}ST_DetectFaceData;
//�ȳ���ԭʼ����
typedef struct tagThermalAVData
{
	char				magic[4];				//ħ���ţ�Ĭ��0xffff ffff
	unsigned int		vesion;					//����Э��汾�ţ������ֽ���
	unsigned int		head_len;				//ͷ���ݳ��ȣ������ֽ���
	unsigned int		payload_len;			//ԭʼ���ݳ��ȣ������ֽ���
	unsigned int		image_width;			//rawͼ�Ŀ������ֽ���
	unsigned int		image_hight;			//rawͼ�ĸߣ������ֽ���
	unsigned int		image_stride;			//rawͼ���ȣ��ֽ�Ϊ��λ, imageStride/imageWidth=ÿ������ռ���ֽ�����Ŀǰrawͼ���Ǹ�14bit��Ч���������ֽ���
	unsigned int		mirror_mode;			//rawͼ����ģʽ��: ��������ˮƽ������ֱ����ˮƽ��ֱ��,�������ֽ���
	unsigned int		capture_timeh;          //rawͼ��ʱ�����λ�������ֽ���
	unsigned int		capture_timel;          //rawͼ��ʱ�����λ�������ֽ���
	unsigned int		sequence_id;			//���ID,�������ֽ���
	char				reserve[64];			//Ԥ���ֶ�
}ST_ThermalAVData;


// Ӳ������
typedef struct _dev_hw_cap_t_
{
	unsigned short				chn_num;	              //ͨ���������������
	unsigned char				audio_in_num;	          //��Ƶ�������
	unsigned char				sound_type;	              //��Ƶͨ������
	unsigned char				audio_out_num;	          //��Ƶ�������
	unsigned char				alarm_in_num;	          //�����������
	unsigned char				alarm_out_num;	          //�����������
	unsigned char				rs485_num;	              //RS485���ڸ���
	unsigned char				rs232_Num;	              //RS232���ڸ���
	unsigned char				netcard_num;	          //������������
	unsigned char				usb_num;	              //USB����
	unsigned char				sd_num;	                  //SD���ĸ���
	unsigned char				hd_num;	                  //Ӳ�̵ĸ���
	unsigned char				is_wifi;	              //�Ƿ�֧��wifi
	unsigned char				is_poe;	                  //�Ƿ�֧��POE
	unsigned char				is_ir;	                  //�Ƿ�֧�ֺ���
	unsigned char				is_pir;	                  //�Ƿ�֧��PIR
	unsigned char				is_bnc;	                  //�Ƿ�֧��ģ��ͨ��
	unsigned char				is_ptz;	                  //�Ƿ�֧��������̨
	unsigned char				is_face;	                //�Ƿ�֧��������
	unsigned short              virtualPTZ_Type;            //����PTZ����
	unsigned short              aiDetec_Type;             //AI ���ץ�����ͣ����������ơ����Σ�����ʶ�𣬸����ͷḻ���������������⣬��hardware������
	unsigned char				is_faceDetect;	           //�Ƿ�֧���������
	unsigned char               resv;                      // �����ֶ�
}dev_hw_cap_t;

// �������
typedef struct _dev_sw_cap_t_
{
	unsigned char				max_user_num;	    //����¼�û���
	unsigned char				max_preview_num;	//���ʵʱԤ��·��
	unsigned char				max_pb_num;	        //���طź�����·��
	unsigned char               resv;               //�����ֶ�
}dev_sw_cap_t;

// ��Ƶ����
typedef struct _dev_audio_cap_t_
{
	unsigned char               is_interphone;       //�Խ�
	unsigned char               is_audio_in;         //��Ƶ����
	unsigned char               is_audio_out;        //��Ƶ���
	unsigned char               resv;                //�����ֶ�
}dev_audio_cap_t;

// �豸��Ҫ��Ϣ
typedef struct _dev_general_info_t_
{
	char	 dev_id[MAX_DEV_NAME_LEN];	                  //�豸Id
	char	 dev_name[MAX_DEV_NAME_LEN];				  //�豸����
	char	 dev_ip[MAX_IP_BUF_LEN];					  //�豸IP
	char	 dev_mac[MAX_IP_BUF_LEN];					  //MAC��ַ
	char	 dev_man_name[MAX_DEV_NAME_LEN];              //��������
	char	 dev_man_id[MAX_DEV_NAME_LEN];                //�豸�ͺ�
	char	 prod_model[MAX_DEV_NAME_LEN];	              //��Ʒģ��
	char     dev_sn[MAX_DEV_NAME_LEN];	                  //SN
	char     sw_info[MAX_DEV_NAME_LEN];	                  //�������Ϣ
	char     hw_info[MAX_DEV_NAME_LEN];	                  //Ӳ����Ϣ
	short    dev_type;                                   //�豸����
	unsigned short dev_port;                          //�豸�˿�
}dev_general_info_t;

// �豸����
typedef struct _dev_name_t_
{
	char dev_name[MAX_DEV_NAME_LEN];				  //�豸����
}dev_name_t;

// �豸ʱ��
typedef struct _dev_time_t_
{
	unsigned short         year;                     //��
	unsigned char          mon;                      //��
	unsigned char          day;                      //��
	unsigned char          hour;                     //ʱ
	unsigned char          min;                      //��
	unsigned char          sec;                      //��
	unsigned char          resv;                     //����
}dev_time_t;

// ntp����
typedef struct _ntp_param_t_
{
	char             serv_ip[MAX_IP_BUF_LEN];              //server ip
	unsigned short   serv_port;                            //server port
	unsigned char    is_enable;                            //�Ƿ�����ntp
	unsigned char    ip_proto_ver;                         //ip Э��汾
	unsigned int     ntp_time;                             //ntp time
}ntp_param_t;

typedef struct _dev_port_t_
{
	char                        dev_id[MAX_DEV_NAME_LEN];             //�豸id
	unsigned short				ctrl_port;	                          //������Ƶ�豸���豸������ƶ˿�
	unsigned short				av_port;	                          //������Ƶ�豸��TCP����Ƶ�˿�
	unsigned short				http_port;	                          //������Ƶ�豸���豸HTTP�˿�
	unsigned short				rtsp_port;	                          //������Ƶ�豸���豸RTSP�˿�
}dev_port_t;

// PTZ ����
typedef enum ptz_operation_e
{
	PTZ_STOP = 0,  //ֹͣ	
	PTZ_UP = 1,        //����
	PTZ_DOWN = 2,      //����	
	PTZ_LEFT = 3,      //��	
	PTZ_RIGHT = 4,     //��		
	PTZ_LEFT_UP = 5,   //����	
	PTZ_LEFT_DOWN = 6, //����
	PTZ_RIGHT_UP = 7,  //����
	PTZ_RIGHT_DOWN = 8, //����
	PTZ_ZOOM_IN = 9,     //����
	PTZ_ZOOM_OUT = 10,   //��Զ	
	PTZ_FOCUS_FAR = 11,  //Զ��
	PTZ_FOCUS_NEAR = 12,  //����	
	PTZ_IRIS_INC = 13,   //��Ȧ���
	PTZ_IRIS_DEC = 14,   //��Ȧ��С
	PTZ_PRESET_SET = 15, //Ԥ��λ����
	PTZ_PRESET_CALL = 16, //Ԥ��λ����
	PTZ_PRESET_DEL = 17,  //Ԥ��λɾ��
	//PTZ_TRACE_SET= 18,   //�켣����
	//PTZ_TRACE_CALL= 19,  //�켣����
	//PTZ_TRACE_DEL= 20,   //�켣ɾ��	
	PTZ_SCAN_CALL=21,    //ɨ�����
	PTZ_SCAN_SET_START = 22,  //����ɨ����ʼ��
	PTZ_SCAN_SET_STOP = 23,  //����ɨ�������
	PTZ_AUTO_FOCUS = 24,     //�Զ��۽�	
	PTZ_AUTO_IRIS = 25,     //�Զ���Ȧ	
	PTZ_START_AUTO_STUDY = 26, //��ʼ��ѧϰ
	PTZ_END_AUTO_STUDY = 27,  //������ѧϰ
	PTZ_RUN_AUTO_STUDY = 28,  //��ѧϰ����
	PTZ_RESET = 29,           //��λ
	PTZ_3D_ORIENTATION = 30,  //��ά���ܶ�λ
	PTZ_TOUR_SET_START = 31,   //����Ѳ����ʼ��
	PTZ_TOUR_ADD_PRESET = 32,  //���Ѳ��Ԥ�õ�
	PTZ_TOUR_SET_END = 33,    //����Ѳ�ν�����
	PTZ_TOUR_RUN = 34,        //����Ѳ��
	PTZ_TOUR_PAUSE = 35,       //��ͣѲ��
	PTZ_TOUR_DEL = 36,        //ɾ��Ѳ��
	PTZ_TOUR_CONTINUE = 200,   //����Ѳ�Σ�����ͣѲ�����ʹ�ã�
	PTZ_KEEPER_SET = 37,      //����λ����
	PTZ_KEEPER_RUN = 38,      //���п���λ
	PTZ_RUN_BRUSH = 39,      //��ˢ����
	PTZ_OPEN_LIGHT = 40,       //�򿪵�
	PTZ_CLOSE_LIGHT = 41,      //�رյ�
	PTZ_SCAN_REMOVE = 44,      //ɾ��ɨ��
	PTZ_REMOVE_AUTO_STUDY = 45,     //ɾ����ѧϰ
	PTZ_INFRARED_CTRL = 46,        //����ƿ���
	PTZ_GET_PTZ_POSTION_REQ = 47,   //�����ȡPTZλ��
	PTZ_GET_PTZ_POSTION_RESP = 48,   //PTZλ��Ӧ��
	PTZ_SET_PTZ_POSTION = 49,        //����PTZλ��
	PTZ_SET_PTZ_NORTH_POSTION = 50,   //��������λ��
	PTZ_GET_PRESET_REQ = 51,      //��ȡԤ��λ����
	PTZ_GET_PRESET_RESP = 52,     //��ȡԤ��λӦ��
	PTZ_GET_TOUR_REQ = 53,    //��ȡѲ������
	PTZ_GET_TOUR_RESP = 54,    //��ȡѲ��Ӧ��
	PTZ_GET_SCAN_REQ = 55,    //��ȡɨ������
	PTZ_GET_SCAN_RESP = 56,    //��ȡɨ��Ӧ��
	PTZ_GET_AUTO_STUDY_REQ = 57,    //��ȡ��ѧϰ����
	PTZ_GET_AUTO_STUDY_RESP = 58,    //��ȡ��ѧϰӦ��
	PTZ_GET_KEEPER_REQ = 59,     //��ȡ����������
	PTZ_GET_KEEPER_RESP = 60,    //��ȡ������Ӧ��
	PTZ_INFRARED_STRL_V2 = 61,   //����ƿ�����չ����
	PTZ_INFRARED_STRL_V2_REQ = 62,   //�������ƿ��Ʋ�������
	PTZ_STOP_BRUSH = 63,       //��ˢֹͣ
	PTZ_360_ROTATE_SCAN = 64,  //360����תɨ��
	PTZ_PERPENDICVULAR_SCAN = 65,  //��ֱɨ��
	PTZ_HEART_BEAT = 66,       //����
	PTZ_INFRARED_CTRL_V2_RESP = 67, //�������ƿ��Ʋ�������Ӧ��
	//PTZ_ZOOM_SPEED= 67,            //��ͷ������Զ�ٶ�ֵ���ã�����Ϊ�뻪Ϊ���ݣ���ֵ������ʱ����������ɳ�ͻ
	PTZ_GET_ALARM_IO_START_REQ = 70,   //�����ȡ����IO״̬
	PTZ_GET_ALARM_IO_START_RESP = 71,  //����IO״̬Ӧ��
	PTZ_PT_STOP_STATUS_RESP = 72,     //PTֹͣ״̬��ѯ
	PTZ_PT_POS_AUTO_RESP = 73,        //�Զ��ϱ�PT����
	PTA_ALARM_IO_STATUS_AUTO_RESP = 74,   //�Զ��ϱ�IO����״̬
	PTZ_GET_ZOOM_VALUE = 75,       //��ͷ�䱶ֵ
	PTZ_GET_PTZ_VERSION = 76,      //��ȡPTZ�汾��
	PTZ_GET_MCU_TEMPERATURE = 77,   //��ȡMCU�¶�
	PTZ_LOAD_DEFAULT = 78,          //�������в���	
	PTZ_GET_PT_POSTION = 79,
	PTZ_SET_VERTICAL_MAX_POSTION = 80,
	PTZ_LENS_RESET = 81,	/*�Զ��۽���ͷ(����ABF)��λ*//*BOOL*/
	PTZ_AUTO_TRACK = 82,
	PTZ_GET_PTZ_ACTION_STATUS_REQ = 83,	//�����ȡPTZ�˶�״̬
	PTZ_GET_PTZ_ACTION_STATUS_RESP = 84,	//PTZ�˶�״̬Ӧ��
	PTZ_SET_WIPER_MODE = 85,	//������ˢģʽ
	PTZ_GET_WIPER_MODE = 86,	//��ȡ��ˢģʽ
	PTZ_SET_PTZ_POWER_SAVE = 87,	//����PTZʡ��
	PTZ_GET_PTZ_POWER_SAVE = 88,	//��ȡPTZʡ��
	PTZ_SET_PT_LIMIT_POS = 89,	//����PT����λ��
	PTZ_GET_PT_LIMIT_POS_REQ = 90,	//��ȡPT����λ������
	PTZ_GET_PT_LIMIT_POS_RESP = 91,	//��ȡPT����λ��Ӧ��
	PTZ_CLEAR_PT_LIMIT_POS = 92,	//���PT����λ��
	PTZ_SET_PT_SELFCHECK = 93,	//����PT�Լ�
	PTZ_GET_PT_SELFCHECK = 94,	//��ȡPT�Լ�
	PTZ_SET_ORIENTATION = 95,	//���ð�װ��ʽ
	PTZ_GET_ORIENTATION = 96,	//��ȡ��װ��ʽ
	PTZ_SET_SHORTCUT = 97,	//���ÿ�ݷ�ʽ
	PTZ_GET_SHORTCUT = 98,	//��ȡ��ݷ�ʽ
	PTZ_SET_DN_MODE = 99,	//������ҹģʽ
	PTZ_SET_WHITE_LIGHT = 100,	//���ð׹��״̬
	PTZ_GET_WHITE_LIGHT = 101,	//��ȡ�׹��״̬
	PTZ_GET_DN_MODE = 102,	//��ȡ��ҹģʽ
	PTZ_SET_ZOOM_VALUE = 103,	//���ñ䱶ֵ
	PTZ_SET_FOCUS_VALUE = 104,	//���þ۽�ֵ
	PTZ_GET_FOCUS_VALUE = 105,	//��ȡ�۽�ֵ
	PTZ_BOW_SCAN = 110,	//����ɨ��	
	PTZ_BOW_SCAN_SET_STARTPOINT = 111,	//���ù���ɨ����ʼ��	
	PTZ_BOW_SCAN_SET_STOPPOINT = 112,	//���ù���ɨ�������	
	PTZ_BOW_SCAN_REMOVE = 113,	//ɾ������ɨ��
	PTZ_BOW_SCAN_PAUSE = 114,	//��ͣ����ɨ��
	PTZ_BOW_SCAN_CONTINUE = 115,	//��������ɨ��
	PTZ_OPEN_DEFOG = 120,	//��͸��
	PTZ_CLOSE_DEFOG = 121	//�ر�͸��   	
}ptz_operation_e;
enum SET_PTZ_POSION_TYPE
{
	POSTION_TYPE_PAN = 0x01, //ˮƽλ�� 
	POSTION_TYPE_TILE = 0x02, //��ֱλ��
	POSTION_TYPE_ZOOM = 0x04  //�Ŵ���
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
	RUN_KEEPER_OFF = 0x00,	//�رտ���λ
	RUN_KEEPER_ON = 0x02  //��������λ
};
enum PTZ_ZOOM
{
	ZOOM_SPEED_MIN = 0x00, //��ͷ������Զ�ٶ�ֵ����Сֵ
	ZOOM_SPEED_MAX = 0x3F  //��ͷ������Զ�ٶ�ֵ�����ֵ
};
enum PTZ_ROTATE_TYPE
{
	ROTATE_TYPE_GEAR = 0x00,	//��λ��1-64��
	ROTATE_TYPE_SPEED = 0x01,	//�ٶ�
	ROTATE_TYPE_DEGREE = 0x02	//����
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

//ʱ�����
typedef struct  _dev_time_zone_param_t_
{
	int				nTimeZone;												//ʱ��

	unsigned char	bDSTOpenFlag;											//����ʱ������־

	int				nBeginMonth;											//����ʱ��ʼ�·�
	int				nBeginWeekly;											//����ʱ��ʼ�ܣ�һ���еĵڼ��ܣ�
	int				nBeginWeekDays;											//���ڼ�
	unsigned int	nBeginTime;												//��ʼʱ��

	int				nEndMonth;												//����ʱ�����·�
	int				nEndWeekly;												//����ʱ�����ܣ�һ���еĵڼ��ܣ�
	int				nEndWeekDays;											//���ڼ�
	unsigned int	nEndTime;												//����ʱ��

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




//�ƻ�ʱ��
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
	int				time_zone;				//ʱ��
	unsigned short	day_light_saving_time;	//����Ӫʱ
	unsigned short	year;					//��
	unsigned short 	month;					//��[1,12]
	unsigned short 	day;					//��[1,31]
	unsigned short 	day_of_week;				//���ڼ�[0,6]
	unsigned short 	hour;					//ʱ[0,23]
	unsigned short 	minute;				//��[0,59]
	unsigned short 	second;				//��[0,59]
	int 			milli_seconds;			//΢��[0,1000000]
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

//��������
typedef struct _alarm_action_
{
	int action_type;	//����Դ����
	int action_id;		//����ԴID
	char action_name[MAX_DEV_NAME_LEN];		//����Դ����
}alarm_action;

//PTZ��������
typedef struct _ptz_action_param_
{
	int ptz_action_type;	//�������ͣ�Ԥ��λ���켣�ȣ�
	int ptz_action_id;		//����ID���û�֮ǰ���õ�Ԥ��λID���켣ID�ȣ�
	int ptz_channel_id;		//PTZͨ��ID
	alarm_action  alarm_act;
}ptz_action_param;

//�����������
typedef struct _alarm_out_param_
{
	char dev_id[MAX_LENGTH_DEVICEID + 1];		//�豸id
	int alarm_out_id;	//��������˿ڵ�ID��
	int alarm_out_flag;	//���������־
	int event_type_id;	//�����¼�����
	int alarm_time;		//�������ʱ��
	alarm_action  alarm_act;
}alarm_out_param;

//����¼�����
typedef struct _record_act_param_
{
	unsigned char pre_record_flag; //�Ƿ���Ԥ¼
	int			delay_record_time;	//��¼��ʱ��
	alarm_action	alarm_act;
}record_act_param;


//������������
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

//IO�����������Ͳ���
typedef struct _io_alarm_insource_para_
{
	alarm_action		alarm_act;
	unsigned char		enable_flag;	//�������
	int					alarm_inval;	//�������
	int					valid_level;	//��Ч��ƽ
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
	unsigned short		enable_flag;	//�Ƿ��������̱���(false���������� true������)
	int		            alarm_inval;	//�ϱ��������λΪ�룬��С���Ϊ10�룬���Ϊ86400��(1��)
	int					alarm_thresold;//������ֵ, ��λΪ�ٷֱ�
	unsigned short		disk_full_enable_flag;
	unsigned short		disk_error_enable_flag;
	unsigned short		no_disk_enable_flag;
	schedule_time_list  schedule_para;
}disk_alarm_source_para;

//���̱�������
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



//��ѯ������Ϣ
typedef struct _qry_info_
{
	char				dev_id[MAX_LENGTH_DEVICEID + 1];		//�豸ID
	long				channel_id;								//ͨ����
	long				record_mode;							//��ѯģʽ(¼���ѯor���ղ�ѯ)
	long				select_mode;							//��ѯģʽ(0:���У�1�������Ͳ�ѯ��2����ʱ���ѯ)
	long				major_type;								//������
	long				minor_type;								//������
	long				precision;								//����
	int					record_segment_interval;		////��ѯ��ʱ�䳤�ȣ�ÿ���ʱ���ȣ�
	time_struct			begin_time;						//��ʼʱ��
	time_struct			end_time;						//����ʱ��

}qry_info;


typedef struct _qry_info_list_
{
	int				qry_info_count;
	qry_info		qry_info_list[CONST_MAX_ALARM_OUT_NUM];
}qry_info_para_list;

typedef struct  _alarm_info_qry_
{
	char			dev_ip[MAX_IP_BUF_LEN];					//�豸IP
	char			dev_id[MAX_LENGTH_DEVICEID + 1];
	int				source_id;	//����ԴId
	int				select_mode;	//��ѯģʽ :SELECT_MODE_ALL
	char			source_name[MAX_DEV_NAME_LEN];	//Դ����
	int				major_type;	//����������
	int				minor_type;	//����������
	unsigned long				alarm_begin_time;	//��ѯ��ʼʱ��
	time_struct				alarm_begin_time_struct;	//
	unsigned long				alarm_end_time;	//��ѯ����ʱ��
	time_struct				alarm_end_time_struct;	//
}alarm_info_qry;


#ifdef WIN32
typedef enum sdks_zoomin_graduate_e
{
	SDKS_ZOOMIN_GRADUATE_MIN = 0, ///<�Ŵ���
	SDKS_ZOOMIN_GRADUATE_1,       ///<�Ŵ��ʼ�0.25 
	SDKS_ZOOMIN_GRADUATE_2,       ///<�Ŵ��ʼ�0.5 
	SDKS_ZOOMIN_GRADUATE_3,       ///<�Ŵ��ʼ�0.75 
	SDKS_ZOOMIN_GRADUATE_4,       ///<�Ŵ��ʼ�1.0 
	SDKS_ZOOMIN_GRADUATE_5,       ///<�Ŵ��ʼ�1.25 
	SDKS_ZOOMIN_GRADUATE_6,       ///<�Ŵ��ʼ�1.5 
	SDKS_ZOOMIN_GRADUATE_7,       ///<�Ŵ��ʼ�1.75 
	SDKS_ZOOMIN_GRADUATE_8,       ///<�Ŵ��ʼ�2.0 
	SDKS_ZOOMIN_GRADUATE_9,       ///<�Ŵ��ʼ�2.25 
	SDKS_ZOOMIN_GRADUATE_10,      ///<�Ŵ��ʼ�2.5 
	SDKS_ZOOMIN_GRADUATE_11,      ///<�Ŵ��ʼ�2.75 
	SDKS_ZOOMIN_GRADUATE_12,      ///<�Ŵ��ʼ�3.0
	SDKS_ZOOMIN_GRADUATE_13,      ///<�Ŵ��ʼ�3.25 
	SDKS_ZOOMIN_GRADUATE_14,      ///<�Ŵ��ʼ�3.5 
	SDKS_ZOOMIN_GRADUATE_15,      ///<�Ŵ��ʼ�3.75 
	SDKS_ZOOMIN_GRADUATE_16,      ///<�Ŵ��ʼ�4.0 
	SDKS_ZOOMIN_GRADUATE_17,      ///<�Ŵ��ʼ�4.25 
	SDKS_ZOOMIN_GRADUATE_18,      ///<�Ŵ��ʼ�4.5 
	SDKS_ZOOMIN_GRADUATE_19,      ///<�Ŵ��ʼ�4.75
	SDKS_ZOOMIN_GRADUATE_20,	  ///<�Ŵ��ʼ�5.0 
	SDKS_ZOOMIN_GRADUATE_21,	  ///<�Ŵ��ʼ�5.25
	SDKS_ZOOMIN_GRADUATE_22,	  ///<�Ŵ��ʼ�5.5 
	SDKS_ZOOMIN_GRADUATE_23,	  ///<�Ŵ��ʼ�5.75
	SDKS_ZOOMIN_GRADUATE_24,	  ///<�Ŵ��ʼ�6.0 
	SDKS_ZOOMIN_GRADUATE_25,	  ///<�Ŵ��ʼ�6.25
	SDKS_ZOOMIN_GRADUATE_26,	  ///<�Ŵ��ʼ�6.5 
	SDKS_ZOOMIN_GRADUATE_27,	  ///<�Ŵ��ʼ�6.75
	SDKS_ZOOMIN_GRADUATE_28,	  ///<�Ŵ��ʼ�7.0 
	SDKS_ZOOMIN_GRADUATE_29,	  ///<�Ŵ��ʼ�7.25
	SDKS_ZOOMIN_GRADUATE_30,	  ///<�Ŵ��ʼ�7.5 
	SDKS_ZOOMIN_GRADUATE_31,	  ///<�Ŵ��ʼ�7.75
	SDKS_ZOOMIN_GRADUATE_32,	  ///<�Ŵ��ʼ�8.0 
	SDKS_ZOOMIN_GRADUATE_33,	  ///<�Ŵ��ʼ�8.25
	SDKS_ZOOMIN_GRADUATE_34,	  ///<�Ŵ��ʼ�8.5 
	SDKS_ZOOMIN_GRADUATE_35,	  ///<�Ŵ��ʼ�8.75
	SDKS_ZOOMIN_GRADUATE_36,	  ///<�Ŵ��ʼ�9.0 
	SDKS_ZOOMIN_GRADUATE_37,	  ///<�Ŵ��ʼ�9.25
	SDKS_ZOOMIN_GRADUATE_38,	  ///<�Ŵ��ʼ�9.5 
	SDKS_ZOOMIN_GRADUATE_39,	  ///<�Ŵ��ʼ�9.75
	SDKS_ZOOMIN_GRADUATE_40,	  ///<�Ŵ��ʼ�10.0 
	SDKS_ZOOMIN_GRADUATE_41,	  ///<�Ŵ��ʼ�10.25
	SDKS_ZOOMIN_GRADUATE_42,	  ///<�Ŵ��ʼ�10.5 
	SDKS_ZOOMIN_GRADUATE_43,	  ///<�Ŵ��ʼ�10.75
	SDKS_ZOOMIN_GRADUATE_44,	  ///<�Ŵ��ʼ�11.0 
	SDKS_ZOOMIN_GRADUATE_45,	  ///<�Ŵ��ʼ�11.25
	SDKS_ZOOMIN_GRADUATE_46,	  ///<�Ŵ��ʼ�11.5 
	SDKS_ZOOMIN_GRADUATE_47,	  ///<�Ŵ��ʼ�11.75
	SDKS_ZOOMIN_GRADUATE_48,	  ///<�Ŵ��ʼ�12.0 
	SDKS_ZOOMIN_GRADUATE_49,	  ///<�Ŵ��ʼ�12.25
	SDKS_ZOOMIN_GRADUATE_50,	  ///<�Ŵ��ʼ�12.5 
	SDKS_ZOOMIN_GRADUATE_51,	  ///<�Ŵ��ʼ�12.75
	SDKS_ZOOMIN_GRADUATE_52,	  ///<�Ŵ��ʼ�13.0 
	SDKS_ZOOMIN_GRADUATE_53,	  ///<�Ŵ��ʼ�13.25
	SDKS_ZOOMIN_GRADUATE_54,	  ///<�Ŵ��ʼ�13.5 
	SDKS_ZOOMIN_GRADUATE_55,	  ///<�Ŵ��ʼ�13.75
	SDKS_ZOOMIN_GRADUATE_56,	  ///<�Ŵ��ʼ�14.0 
	SDKS_ZOOMIN_GRADUATE_57,	  ///<�Ŵ��ʼ�14.25
	SDKS_ZOOMIN_GRADUATE_58,	  ///<�Ŵ��ʼ�14.5 
	SDKS_ZOOMIN_GRADUATE_59,	  ///<�Ŵ��ʼ�14.75
	SDKS_ZOOMIN_GRADUATE_60,	  ///<�Ŵ��ʼ�15.0 
	SDKS_ZOOMIN_GRADUATE_MAX
}sdks_zoomin_graduate_e;

#endif

#endif
