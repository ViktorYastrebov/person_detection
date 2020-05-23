#ifndef  __SDKS_FACE_H__
#define  __SDKS_FACE_H__
#include "sdk_def.h"
#include "rj_type.h"

typedef struct rename_t
{
	char  old_name[64];
	char  new_name[64];
}rename_t;

typedef struct rename_info_t
{
	rename_t  name[10];
	int       data_chn;
	int       num;
}rename_info;

//typedef struct jy_face_param_t
//{
//	uint16		 name_len;
//	char		 name[32];//人脸名称
//	uint16		 id_len;
//	char		 id[32];
//	char		 gender;
//	uint32		 birthday;
//	uint16		 group_len;
//	char         group[32];//人脸所属库
//	uint16		 type_len;
//	char		 type[32];//人脸类型
//	uint32		 s_time;
//	uint32		 e_time;
//}jy_face_param;



typedef struct face_search_param_t
{
	uint16		 chn_num;//通道数
	uint16		 chn_list;//通道数组
	uint32		 s_time;//检索范围起始时间
	uint32		 e_time;//检索范围结束时间
	char         search_type;
	char		 similarity;//相似度
	uint16		 id_len;//特征id
	char		 id[32];
	uint16		 name_len;
	char		 name[32];//特征名字
	char         gender;//特征性别
}face_search_param;


typedef struct page_param_t
{
	int			 total_pic_num;//图片索引总数
	int			 total_page_num;//客户端分页总数
	int			 page;//第几页
}page_param;

typedef struct  group_page_param_t
{
	char	groups[32][64];//客户端
	int		total_mem_num;
	int		total_page_num;
	int		page;//当前第几页
	int     group_num;
}group_page_param;


SDKS_API int sdks_start_face(unsigned int handle, SDK_FACE_CB face_cb, void *p_obj);
SDKS_API int sdks_stop_face(unsigned int handle);

//获取人脸库请求
SDKS_API int sdks_face_get_group(unsigned int handle, char** result);
//增加人脸库
SDKS_API int sdks_face_add_group(unsigned int handle, char *p_db_info);
//删除人脸库
SDKS_API int sdks_face_del_group(unsigned int handle, char *p_db_info);
//重命名人脸库
SDKS_API int sdks_face_rename_group(unsigned int handle, char *p_db_info);

//获取人脸库类型请求
SDKS_API int sdks_face_get_group_type(unsigned int handle, char** result);
//增加人脸类型
SDKS_API int sdks_face_add_group_type(unsigned int handle, char *p_db_info);
//删除人脸类型
SDKS_API int sdks_face_del_group_type(unsigned int handle, char *p_db_info);

//添加人脸图片到人脸库
SDKS_API int sdks_add_face_data_to_group(unsigned int handle, char *p_param, char *pic_data, int pic_size); 
//删除人脸图片数据
SDKS_API int sdks_del_face_data(unsigned int handle,char *p_param); 
//修改人脸图片
SDKS_API int  sdks_mod_face_data(unsigned int handle, char *p_param, char *pic_data, int pic_size); 
//查询底库图片
SDKS_API int  sdks_get_face_all_node(unsigned int handle,char *p_param, char **result);
SDKS_API int  sdks_get_face_by_node(unsigned int handle, char *p_info);
//按人脸库索引查询人脸信息
 SDKS_API int  sdks_get_face_info(unsigned int handle, char *p_param, char **p_result);

//实时库条件查询人脸
 SDKS_API int sdks_dev_get_database_index(unsigned int handle, int chn, char *p_buf, int size, char *param, int *p_task, char **p_result);
 SDKS_API int sdks_dev_get_database_info(unsigned int handle, char *param);
#endif