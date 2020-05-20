///////////////////////////////////////////////////////////////////////////
//  Copyright(c) 2015-2017, All Rights Reserved
//  Created: 2016/01/05
//
/// @file    sdks.h
/// @brief   sdk接口
/// @author
/// @version 0.3
/// @warning 没有警告
///////////////////////////////////////////////////////////////////////////
#ifndef __SDKS_H__
#define __SDKS_H__
#include "sdk_def.h"

///< sdks初始化/退出
SDKS_API int   sdks_dev_init(const char* p_json_setup_in);
SDKS_API void  sdks_dev_quit();
SDKS_API void  sdks_free_result(void* p_result);

///< 连接，返回handle


SDKS_API unsigned int  sdks_dev_conn(const char* p_ip, unsigned short  port, const char*  p_user, const char* p_passwd, SDK_DISCONN_CB disconn_cb, void* p_obj);
//ssl encryption connect
SDKS_API unsigned int  sdks_dev_conn_ssl(const char* p_ip, unsigned short  port, const char*  p_user, const char* p_passwd, SDK_DISCONN_CB disconn_cb, void* p_obj);
SDKS_API int sdks_dev_get_con_sta(unsigned int handle);

///< 异步连接接口
SDKS_API int  sdks_dev_conn_async(const char* p_ip, unsigned short  port, const char*  p_user, const char* p_passwd, SDK_DISCONN_CB disconn_cb, SDK_CONNECT_CB conn_cb, void* p_obj);
SDKS_API void  sdks_dev_conn_close(unsigned int handle);

// Live
SDKS_API int sdks_dev_addr_req(unsigned int handle, int ipprotover, char** p_result);
SDKS_API int sdks_dev_live_start(unsigned int handle, int chn, int stream_type, SDK_STREAM_CB stream_cb, void* p_obj);
SDKS_API int sdks_dev_live_stop(unsigned int handle, int stream_id);
SDKS_API int sdks_dev_chg_stream(unsigned int handle, int stream_id, int new_stream_type);
SDKS_API int sdks_get_video_param(unsigned int handle, int chn, char** p_result);
SDKS_API int sdks_set_video_param(unsigned int handle, char* p_video_param);
SDKS_API int sdks_dev_video_control(unsigned int handle, char* p_audio_para);
//Audio
SDKS_API int sdks_dev_audio_start(unsigned int handle, int stream_id);
SDKS_API int sdks_dev_audio_stop(unsigned int handle, int stream_id);

//Snap
SDKS_API int sdks_get_snap_data(unsigned int handle, char* p_snap_param, char  **p_buf, int *len);
SDKS_API int sdks_get_snap_picture(unsigned int handle, char* p_snap_param);
SDKS_API int sdks_open_snap(unsigned int handle, char* p_snap_param);
SDKS_API int sdks_close_snap(unsigned int handle, char* p_snap_param);
// PlayBack
//s_date和e_date 的格式必须为 "yyyy-mm-dd"
SDKS_API int sdks_dev_pb_date_list(unsigned int handle, int chn, const char* s_date, const char* e_date, char** p_result);
//p_date 的格式必须为 "yyyy-mm-dd"
SDKS_API int sdks_dev_pb_chns_in_date(unsigned int handle, const char* p_date, char** p_result); //v2
// 获取某个通道录像的时间段
SDKS_API int sdks_dev_pb_get_rec_list(unsigned int handle, int chn, const char* p_date, char** p_result);
// s_time和e_time 的格式必须为 "yyyy-mm-dd HH:mm:ss"
SDKS_API int sdks_dev_pb_start(unsigned int handle, int chn, int stream_type, const char* s_time, const char* e_time, SDK_STREAM_CB stream_cb, void* p_obj);
SDKS_API int sdks_dev_pb_seek(unsigned int handle, int stream_id, const char* time);
SDKS_API int sdks_dev_pb_pause(unsigned int handle, int stream_id);
SDKS_API int sdks_dev_pb_resume(unsigned int handle, int stream_id);
SDKS_API int sdks_dev_pb_stop(unsigned int handle, int stream_id);
SDKS_API int sdks_get_pb_video_param(unsigned int handle, char** p_result);
SDKS_API int sdks_set_pb_video_speed(unsigned int handle, int stream_id, int rate);
// Record
SDKS_API int sdks_dev_open_rec(const char* p_path);
SDKS_API int sdks_dev_record(int record_id, ST_AVFrameData* p_frame);
SDKS_API int sdks_dev_stop_rec(int record_id);

// Alarm
SDKS_API int sdks_dev_start_alarm(unsigned int handle, SDK_ALARM_CB alarm_cb, void* p_obj);
SDKS_API int sdks_dev_stop_alarm(unsigned int handle);

//IO报警
SDKS_API int sdks_dev_get_io_alarm_event(unsigned int handle, int chn, int alarm_source_id, char** p_result);
SDKS_API int sdks_dev_set_io_alarm_para(unsigned int handle, const io_alarm_event_para_list* p_io_alarm_para);
SDKS_API int sdks_dev_json_set_io_alarm_para(unsigned int handle, const char* p_io_alarm_para);

//磁盘报警 
SDKS_API int sdks_dev_json_set_disk_alarm_para(unsigned int handle, const char* p_disk_alarm_para);
SDKS_API int sdks_dev_set_disk_alarm_para(unsigned int handle, const disk_alarm_event_para_list* p_disk_alarm_list);
SDKS_API int sdks_dev_get_disk_alarm_para(unsigned int handle, int chn, char** p_result);

//查询报警符合要求的日期请求列表,s_time和e_time 的格式必须为 "yyyy-mm-dd HH:mm:ss"
SDKS_API int sdks_dev_get_match_alarm_date_list(unsigned int handle, const qry_info_para_list * p_qry_info, char** p_result);
SDKS_API int sdks_dev_json_get_match_alarm_date_list(unsigned int handle, const char* p_qry_info, const char* s_time, const char* e_time, char** p_result);

//获取告警设备的信息列表请求
SDKS_API int sdks_dev_get_alarm_camera_info_list(unsigned int handle, const alarm_info_qry* p_alarm_info_qry, char** p_result);
SDKS_API int sdks_dev_json_get_alarm_camera_info_list(unsigned int handle, const char* s_time, const char* e_time, const char* p_alarm_info_qry, char** p_result);

// s_time和e_time 的格式必须为 "yyyy-mm-dd HH:mm:ss"
SDKS_API int sdks_dev_get_alarm_list(unsigned int handle, int chn, const char* s_time, const char* e_time, char** p_result);
//手动报警
SDKS_API int sdks_dev_manual_alarmout(unsigned int handle, int chn, const int alarmout_id, int control_flag);

//录像策略
SDKS_API int sdks_dev_get_record_policy(unsigned int handle, int chn, int record_mode, char** p_result);
SDKS_API int sdks_dev_set_record_policy(unsigned int handle, int chn, char* p_record_para);

//录像状态
SDKS_API int sdks_dev_get_record_state(unsigned int handle, int chn, char** p_result);
//time的格式必须为 "yyyy-mm-dd HH:mm:ss"
SDKS_API int sdks_dev_get_last_record_time(unsigned int handle, const char* s_time, const char* e_time, char* p_qry_info, char** p_result);

//WIFI信号上报

SDKS_API int sdks_dev_open_wifi_push(unsigned int handle, SDK_WIFI_CB alarm_cb, void* p_obj);
SDKS_API int sdks_dev_close_wifi_push(unsigned int handle);
 
//PTZ    //v2
SDKS_API int sdks_dev_open_ptz(unsigned int handle); 
SDKS_API int sdks_dev_close_ptz(unsigned int handle); 
SDKS_API int sdks_dev_ptz_stop(unsigned int handle, int chn);
SDKS_API int sdks_dev_ptz_rotate(unsigned int handle, int chn, int operation, int speed);
SDKS_API int sdks_dev_ptz_zoom(unsigned int handle, int chn, int operation, int speed);
SDKS_API int sdks_dev_ptz_focus(unsigned int handle, int chn, int operation);
SDKS_API int sdks_dev_ptz_iris(unsigned int handle, int chn, int operation);
SDKS_API int sdks_dev_ptz_preset(unsigned int handle, int chn, int id, int operation);
SDKS_API int sdks_dev_ptz_track(unsigned int handle, int chn, int id, int operation);
SDKS_API int sdks_dev_ptz_scan(unsigned int handle, int chn, int id, int operation);
SDKS_API int sdks_dev_ptz_tour(unsigned int handle, int chn, int id, int operation, int speed,int time);
SDKS_API int sdks_dev_ptz_keeper(unsigned int handle, int chn, int operation, int enable, int type, int id, int time);
SDKS_API int sdks_dev_ptz_threeDimensionalPos(unsigned int handle, int chn, int nX, int nY, int nZoomaTate);
SDKS_API int sdks_dev_ptz_brush(unsigned int handle, int chn, int operation, int mode, int waittime);
SDKS_API int sdks_dev_ptz_light(unsigned int handle, int chn, int operation);
SDKS_API int sdks_dev_ptz_defog(unsigned int handle, int chn, int operation);
SDKS_API int sdks_dev_ptz_postion(unsigned int handle, int chn, int operation, int type, int p_nPan, int p_nTilt, int p_nZoom);
SDKS_API int sdks_dev_get_ptz_postion(unsigned int handle, int chn, char** p_result);
SDKS_API int sdks_dev_get_ptz_req(unsigned int handle, int chn, char** p_result); 
SDKS_API int sdks_dev_set_ptz_speed(unsigned int handle, int chn, int speed); 
SDKS_API int sdks_dev_get_ptz_configue(unsigned int handle, int chn, int operation, char** p_result); 
SDKS_API int sdks_dev_get_ptz_timer(unsigned int handle, int chn, char** p_result);
SDKS_API int sdks_dev_set_ptz_timer(unsigned int handle, int chn, char* p_param);

// 设备能力
SDKS_API int sdks_dev_get_hw_cap(unsigned int handle, dev_hw_cap_t* p_hw_cap);//v2
SDKS_API int sdks_dev_get_hw_cap_by_chn(unsigned int handle, dev_hw_cap_t* p_hw_cap, int chn);//

SDKS_API int sdks_dev_json_get_hw_cap(unsigned int handle, char** p_result);//v2
SDKS_API int sdks_dev_get_sw_cap(unsigned int handle, dev_sw_cap_t* p_sw_cap);//v2
SDKS_API int sdks_dev_json_get_sw_cap(unsigned int handle, char** p_result);//v2
SDKS_API int sdks_dev_get_nw_cap(unsigned int handle, int chn, char** p_result);//v2
SDKS_API int sdks_dev_get_video_cap(unsigned int handle, int chn, char** p_result);
SDKS_API int sdks_dev_get_nvr_cap(unsigned int handle, char** p_result);
SDKS_API int sdks_dev_get_language_cap(unsigned int handle, int chn, char **p_result); //v2
//不同语言对应不同的时区能力列表。
SDKS_API int sdks_dev_get_time_zone_cap(unsigned int handle, int chn, int language_id, char** p_result); //v2
SDKS_API int sdks_dev_get_audio_cap(unsigned int handle, dev_audio_cap_t* p_audio_cap);//v2
SDKS_API int sdks_dev_json_get_audio_cap(unsigned int handle, char** p_result);//v2
SDKS_API int sdks_dev_get_ptz_cap(unsigned int handle, int chn, char** p_result);
SDKS_API int sdks_dev_get_osd_cap(unsigned int handle, int chn, char** p_result);

// 设备基本参数
SDKS_API int sdks_dev_get_general_info(unsigned int handle, dev_general_info_t* p_gene_info); //v2
SDKS_API int sdks_dev_get_dev_name(unsigned int handle, dev_name_t* p_dev_name); //v2
SDKS_API int sdks_dev_set_dev_name(unsigned int handle, dev_name_t* p_dev_name); //v2
SDKS_API int sdks_dev_get_dev_time(unsigned int handle, dev_time_t* p_dev_time); //v2
SDKS_API int sdks_dev_set_dev_time(unsigned int handle, dev_time_t* p_dev_time);//v2
SDKS_API int sdks_dev_json_get_general_info(unsigned int handle, char** p_result); //v2
SDKS_API int sdks_dev_json_get_dev_name(unsigned int handle, char** p_result); //v2
SDKS_API int sdks_dev_json_set_dev_name(unsigned int handle, char* p_param); //v2
SDKS_API int sdks_dev_json_get_dev_time(unsigned int handle, char** p_result);//v2
SDKS_API int sdks_dev_json_set_dev_time(unsigned int handle, char* p_param);//v2
SDKS_API int sdks_dev_get_video_system(unsigned int handle, char** p_result);//v2
SDKS_API int sdks_dev_set_video_system(unsigned int handle, char* p_param);//v2

//NTP自动校时
SDKS_API int sdks_dev_get_dev_ntp(unsigned int handle, ntp_param_t* p_ntp_param);//v2
SDKS_API int sdks_dev_set_dev_ntp(unsigned int handle, ntp_param_t* p_ntp_param);//v2
SDKS_API int sdks_dev_json_get_dev_ntp(unsigned int handle, char** p_result);//v2
SDKS_API int sdks_dev_json_set_dev_ntp(unsigned int handle, char* p_param);//v2
SDKS_API int sdks_dev_get_dev_id(unsigned int handle, int chn, char** p_result);//v2
SDKS_API int sdks_dev_set_dev_id(unsigned int handle, int chn, char* p_dev_id);//v2
SDKS_API int sdks_dev_get_dev_port(unsigned int handle, dev_port_t* p_dev_port);//v2
SDKS_API int sdks_dev_set_dev_port(unsigned int handle, dev_port_t* p_dev_port);//v2
SDKS_API int sdks_dev_json_get_dev_port(unsigned int handle, char** p_result);//v2
SDKS_API int sdks_dev_json_set_dev_port(unsigned int handle, char* p_param);//v2
SDKS_API int sdks_dev_get_dev_language(unsigned int handle, int chn, char** p_result); //v2
SDKS_API int sdks_dev_set_dev_language(unsigned int handle, int chn, int language_id); //v2
//timezone时区
SDKS_API int sdks_get_dev_time_zone(unsigned int handle, char** result); //v2
SDKS_API int sdks_set_dev_time_zone(unsigned int handle, char* p_dev_time); //v2
SDKS_API int sdks_dev_get_p2p_para(unsigned int handle, int chn, char** p_result); //v2
//报警推送
SDKS_API int sdks_dev_set_alarm_push_para(unsigned int handle, char* p_alarm_push_para);
SDKS_API int sdks_dev_delete_alarm_push_para(unsigned int handle, char* p_alarm_push_para);
//安全参数
SDKS_API int sdks_dev_get_security_para(unsigned int handle, int chn, char** p_result); //v2
SDKS_API int sdks_dev_set_security_para(unsigned int handle, int web_mode, unsigned char encrypt_enable); //v2
SDKS_API int sdks_dev_get_nvr_channel_name(unsigned int handle, int chn, char** p_result); //v2
SDKS_API int sdks_dev_get_chn_info(unsigned int handle, char** p_result); //v2
// 网络参数
SDKS_API int sdks_dev_get_net_param(unsigned int handle, char** p_result); //v2
SDKS_API int sdks_dev_set_net_param(unsigned int handle, char* p_net_param); //v2
SDKS_API int sdks_dev_get_ddns(unsigned int handle, char** p_result); //v2
SDKS_API int sdks_dev_set_ddns(unsigned int handle, char* p_net_ddns); //v2
SDKS_API int sdks_dev_get_ddns_provider(unsigned int handle, char** p_result); //v2
//FTP参数
SDKS_API int sdks_dev_get_ftp(unsigned int handle, char** p_result); //v2
SDKS_API int sdks_dev_set_ftp(unsigned int handle, char* p_net_ftp); //v2
//SMTP参数
SDKS_API int sdks_dev_get_smtp(unsigned int handle, char** p_result); //v2
SDKS_API int sdks_dev_set_smtp(unsigned int handle, char* p_net_smtp); //v2
SDKS_API int sdks_dev_get_mtu(unsigned int handle, int* p_mtu); //v2
SDKS_API int sdks_dev_set_mtu(unsigned int handle, int mtu); //v2
//802.1x参数
SDKS_API int sdks_dev_get_8021x(unsigned int handle, char **p_result); //v2
SDKS_API int sdks_dev_set_8021x(unsigned int handle, char *p_param); //v2
//PPPOE参数
SDKS_API int sdks_dev_get_pppoe(unsigned int handle, char **p_result); //v2
SDKS_API int sdks_dev_set_pppoe(unsigned int handle, char *p_param); //v2
//端口映射参数
SDKS_API int sdks_dev_get_port_mapping(unsigned int handle, char **p_result); //v2
SDKS_API int sdks_dev_set_port_mapping(unsigned int handle, char *p_param); //v2
//IP过滤参数
SDKS_API int sdks_get_ip_filter_param(unsigned int handle, char** p_result); //v2
SDKS_API int sdks_set_ip_filter_param(unsigned int handle, char *p_ip_param); //v2
//安全协议参数
SDKS_API int sdks_get_protocol_security_param(unsigned int handle, char** p_result); //v2
SDKS_API int sdks_set_protocol_security_param(unsigned int handle, char* p_param); //v2


//OSD 参数
SDKS_API int sdks_get_osd_param(unsigned int handle, char** p_result);
SDKS_API int sdks_set_osd_param(unsigned int handle, char* p_osd_param);

//隐私遮蔽参数
SDKS_API int sdks_get_blind_param(unsigned int handle, const int chn, char** result);
SDKS_API int sdks_set_blind_param(unsigned int handle, char* p_blind_param);

//伴随流参数
SDKS_API int sdks_get_svc_stream_para(unsigned int handle, int chn, int stream_id, char** p_result);


//ROI 参数
SDKS_API int sdks_get_roi_param(unsigned int handle, char** result);
SDKS_API int sdks_set_roi_param(unsigned int handle, int channel, int stream, char* p_mot_param);

//移动侦测
SDKS_API int sdks_get_mot_param(unsigned int handle, char** result);
SDKS_API int sdks_set_mot_param(unsigned int handle, char *p_mot_param);


//设备搜索
SDKS_API int sdks_get_dev_list(char **p_json_out);


//修改密码
SDKS_API int sdks_modify_password_param(unsigned int handle, char* p_system_user_param);
//创建登录用户名密码
SDKS_API int sdks_create_login_password_param(unsigned int handle, char* p_creat_login_password_param);

//用户权限管理
SDKS_API int sdks_operator_privilege_user(unsigned int handle, int chn, const char* p_user_list, char** p_result);

//sensor
SDKS_API int sdks_reset_sensor_to_last_param(unsigned int handle, int chn);
SDKS_API int sdks_set_sensor_para(unsigned int handle, char* p_sensor_para);


//重启设备
SDKS_API int sdks_dev_reboot(unsigned int handle, int chn);
SDKS_API int sdks_dev_reset(unsigned int handle, int chn);

//热成像接口
SDKS_API int sdks_dev_get_thermal_cap(unsigned int handle, int channel, char **p_result);

SDKS_API int sdks_set_thermal_param(unsigned int handle, int channel, char* p_param);
SDKS_API int sdks_get_thermal_param(unsigned int handle, int channel, char** p_result);

SDKS_API int sdks_set_thermal_area_temperature_measure(unsigned int handle, char* p_param);
SDKS_API int sdks_get_thermal_area_temperature_measure(unsigned int handle, char* p_param, char** p_result);

SDKS_API int sdks_get_thermal_area_feature_temperature(unsigned int handle, char *p_param, char** p_result);//

SDKS_API int sdks_get_thermal_one_point_temperature(unsigned int handle, int channel, int x, int y, char** p_result);
SDKS_API int sdks_get_thermal_any_point_temperature(unsigned int handle, int channel, char *p_param, char** p_result);//返回 -661

SDKS_API int sdks_get_map_relation(unsigned int handle, int  channel, char** p_result);
SDKS_API int sdks_set_map_relation(unsigned int handle, char* p_param);

SDKS_API int sdks_get_temperature_calibration(unsigned int handle, int  channel, char** p_result);
SDKS_API int sdks_set_temperature_calibration(unsigned int handle, int channel, char *p_param);

SDKS_API int sdks_get_thermal_version(unsigned int handle, int  channel, char** p_result);

SDKS_API int sdks_test_thermal_bad_point_correct(unsigned int handle, char* p_param);
SDKS_API int sdks_set_thermal_bad_point_correct(unsigned int handle, int channel);
SDKS_API int sdks_reset_thermal_bad_point_correct(unsigned int handle, int channel);

SDKS_API int sdks_get_thermal_alarm_linkage_param(unsigned int handle, int channel, char **p_result);
SDKS_API int sdks_set_thermal_alarm_linkage_param(unsigned int handle, int channel, char *p_param);

//热成像原始数据流
SDKS_API int sdks_dev_thermal_live_start(unsigned int handle, int chn, int stream_type, SDK_DETECT_CB stream_cb, void* p_obj);
SDKS_API int sdks_dev_thermal_live_stop(unsigned int handle, int stream_id);

//人脸接口
SDKS_API int sdks_get_face_detect_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_face_detect_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_dev_face_detect_start(unsigned int handle, int chn, int stream_type, int type, SDK_DETECT_CB detect_cb, void* p_obj);
SDKS_API int sdks_dev_face_detect_stop(unsigned int handle, int stream_id);
SDKS_API int sdks_dev_face_get_group_num(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_dev_face_get_member(unsigned int handle, int chn, char *p_param, char **p_result);
SDKS_API int sdks_dev_face_check_data(unsigned int handle, int chn, char *p_param, char **p_result);
SDKS_API int sdks_dev_face_get_statis(unsigned int handle, int chn, char *p_param, char **p_result);
SDKS_API int sdks_dev_face_get_attendance_data(unsigned int handle, int chn, char *p_param, char *path_file);


//车牌
SDKS_API int sdks_get_lpr_link_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_lpr_link_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_lpr_ipfilter_list_add(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_lpr_ipfilter_list_delete(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_lpr_ipfilter_list_modify(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_lpr_ipfilter_list_num(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_get_lpr_ipfilter_list(unsigned int handle, int chn, char *p_param, char **p_result);
SDKS_API int sdks_lpr_ipfilter_list_search_open(unsigned int handle, int chn, char *p_param, char **p_result);
SDKS_API int sdks_lpr_ipfilter_list_search_get(unsigned int handle, int chn, char *p_param, char **p_result);
SDKS_API int sdks_lpr_ipfilter_list_search_close(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_lpr_ipfilter_list_file_download(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ai_multi_object_detect_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ai_multi_object_detect_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ai_multi_object_detect_ability(unsigned int handle, int chn, char **p_result);

//设备日志
SDKS_API int sdks_get_device_log(unsigned int handle, char *p_param, char **p_result);

//智能分析
SDKS_API int sdks_get_ia_version(unsigned int handle, int chn, char **p_result); //版本信息
SDKS_API int sdks_get_ia_perimeter_ability(unsigned int handle, int chn, char **p_result); //入侵能力
SDKS_API int sdks_get_ia_svf_ability(unsigned int handle, int chn, char **p_result); //警戒线能力
SDKS_API int sdks_get_ia_dvf_ability(unsigned int handle, int chn, char **p_result); //双警戒线能力
SDKS_API int sdks_get_ia_loiter_ability(unsigned int handle, int chn, char **p_result); //徘徊能力
SDKS_API int sdks_get_ia_multi_loiter_ability(unsigned int handle, int chn, char **p_result); //多人徘徊能力
SDKS_API int sdks_get_ia_object_left_ability(unsigned int handle, int chn, char **p_result); //物品遗留能力
SDKS_API int sdks_get_ia_object_removed_ability(unsigned int handle, int chn, char **p_result); //物品移走能力
SDKS_API int sdks_get_ia_abnormal_speed_ability(unsigned int handle, int chn, char **p_result); //异常速度能力
SDKS_API int sdks_get_ia_converse_ability(unsigned int handle, int chn, char **p_result); //逆行能力
SDKS_API int sdks_get_ia_legal_parking_ability(unsigned int handle, int chn, char **p_result); //非法停车能力
SDKS_API int sdks_get_ia_signal_bad_ability(unsigned int handle, int chn, char **p_result); //视频信号异常能力
SDKS_API int sdks_get_ia_advanced_ability(unsigned int handle, int chn, char **p_result); //高级配置能力
SDKS_API int sdks_get_ia_perimeter_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_perimeter_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_svf_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_svf_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_dvf_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_dvf_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_loiter_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_loiter_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_multi_loiter_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_multi_loiter_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_object_left_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_object_left_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_object_removed_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_object_removed_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_abnormal_speed_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_abnormal_speed_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_converse_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_converse_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_legal_parking_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_legal_parking_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_signal_bad_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_signal_bad_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_ia_advanced_param(unsigned int handle, int chn, char **p_result);
SDKS_API int sdks_set_ia_advanced_param(unsigned int handle, int chn, char *p_param);

//鱼眼
SDKS_API int sdks_get_fisheye_ability(unsigned int handle, int chn, char **p_result); 
SDKS_API int sdks_get_fisheye_param(unsigned int handle, int chn, char **p_result); 
SDKS_API int sdks_set_fisheye_param(unsigned int handle, int chn, char *p_param);
SDKS_API int sdks_get_fisheye_video_layout(unsigned int handle, int chn, char **p_result);

//语音对讲
SDKS_API int sdks_open_microphone(unsigned int handle, SDK_STREAM_CB microphone_cb, void *p_obj);
SDKS_API int sdks_close_microphone(unsigned int handle);
SDKS_API int sdks_dev_send_audio_data(unsigned int handle, char *p_data, int audio_len);

//升级设备
SDKS_API int sdks_update_nvr(unsigned int handle, char *p_path);


//////////////////////////////////////////////////////////////////////////
#endif // __SDKS_H__
//end
