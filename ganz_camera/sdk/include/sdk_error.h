#ifndef __SDK_ERROR_H__
#define __SDK_ERROR_H__

///< ������
typedef enum sdks_error_e
{
	SDKS_ERR_OK,
	SDKS_ERROR_UNKNOWN = 1,                        ///< δ֪����
	SDKS_ERROR_SERVER_BUSY = 2,                    ///< ������æ
	SDKS_ERROR_NOT_SUPPORT = 3,                    ///< ��֧��
	SDKS_ERROR_PARAMETER = 4,                      ///< ��������
	SDKS_ERROR_REQUEST_TIMEOUT = 5,                ///< ����ʱ
	SDKS_ERROR_DEV_AUTH_CODE = 3100,               ///< ��Ȩ�����,��Ȩʧ��
	SDKS_ERROR_DEV_BIND_TO_DEV = 3101,             ///< �豸�Ѱ󶨵������豸
	SDKS_ERROR_DEV_BIND_TO_ACCOUNT = 3102,         ///< �豸�Ѱ󶨵��˻�
	SDKS_ERR_NO_FRAME = 10000,                    ///< renderû��ˢ����Ƶ֡
	SDKS_ERR_UNREACHABLE,		                  ///< �豸���ɴ�
	SDKS_ERR_CONNECT_TIMEOUT,	                  ///< �����豸��ʱ
	SDKS_ERR_DISCONNECT,		                  ///< ���ӶϿ�
	SDKS_ERR_DECODE,			                  ///< ��������̫��
	SDKS_ERR_REC,				                  ///< ¼��̫��
	SDKS_ERR_CAPTURE,			                  ///< ץͼʧ��
	SDKS_ERR_VIEW,				                  ///< ��̫�ര��
	SDKS_ERR_NOT_OPEN_VIEW,	                      ///< û�д򿪴���
	SDKS_ERR_TOO_MANY_STREAM,	                  ///< �򿪵���̫��
}sdks_error_e;

#endif
