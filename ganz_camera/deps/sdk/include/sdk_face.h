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
//	char		 name[32];//��������
//	uint16		 id_len;
//	char		 id[32];
//	char		 gender;
//	uint32		 birthday;
//	uint16		 group_len;
//	char         group[32];//����������
//	uint16		 type_len;
//	char		 type[32];//��������
//	uint32		 s_time;
//	uint32		 e_time;
//}jy_face_param;



typedef struct face_search_param_t
{
	uint16		 chn_num;//ͨ����
	uint16		 chn_list;//ͨ������
	uint32		 s_time;//������Χ��ʼʱ��
	uint32		 e_time;//������Χ����ʱ��
	char         search_type;
	char		 similarity;//���ƶ�
	uint16		 id_len;//����id
	char		 id[32];
	uint16		 name_len;
	char		 name[32];//��������
	char         gender;//�����Ա�
}face_search_param;


typedef struct page_param_t
{
	int			 total_pic_num;//ͼƬ��������
	int			 total_page_num;//�ͻ��˷�ҳ����
	int			 page;//�ڼ�ҳ
}page_param;

typedef struct  group_page_param_t
{
	char	groups[32][64];//�ͻ���
	int		total_mem_num;
	int		total_page_num;
	int		page;//��ǰ�ڼ�ҳ
	int     group_num;
}group_page_param;


SDKS_API int sdks_start_face(unsigned int handle, SDK_FACE_CB face_cb, void *p_obj);
SDKS_API int sdks_stop_face(unsigned int handle);

//��ȡ����������
SDKS_API int sdks_face_get_group(unsigned int handle, char** result);
//����������
SDKS_API int sdks_face_add_group(unsigned int handle, char *p_db_info);
//ɾ��������
SDKS_API int sdks_face_del_group(unsigned int handle, char *p_db_info);
//������������
SDKS_API int sdks_face_rename_group(unsigned int handle, char *p_db_info);

//��ȡ��������������
SDKS_API int sdks_face_get_group_type(unsigned int handle, char** result);
//������������
SDKS_API int sdks_face_add_group_type(unsigned int handle, char *p_db_info);
//ɾ����������
SDKS_API int sdks_face_del_group_type(unsigned int handle, char *p_db_info);

//�������ͼƬ��������
SDKS_API int sdks_add_face_data_to_group(unsigned int handle, char *p_param, char *pic_data, int pic_size); 
//ɾ������ͼƬ����
SDKS_API int sdks_del_face_data(unsigned int handle,char *p_param); 
//�޸�����ͼƬ
SDKS_API int  sdks_mod_face_data(unsigned int handle, char *p_param, char *pic_data, int pic_size); 
//��ѯ�׿�ͼƬ
SDKS_API int  sdks_get_face_all_node(unsigned int handle,char *p_param, char **result);
SDKS_API int  sdks_get_face_by_node(unsigned int handle, char *p_info);
//��������������ѯ������Ϣ
 SDKS_API int  sdks_get_face_info(unsigned int handle, char *p_param, char **p_result);

//ʵʱ��������ѯ����
 SDKS_API int sdks_dev_get_database_index(unsigned int handle, int chn, char *p_buf, int size, char *param, int *p_task, char **p_result);
 SDKS_API int sdks_dev_get_database_info(unsigned int handle, char *param);
#endif