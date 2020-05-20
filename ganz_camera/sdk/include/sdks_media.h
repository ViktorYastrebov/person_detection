///////////////////////////////////////////////////////////////////////////
//  Copyright(c) 2015-2018, All Rights Reserved
//  Created: 2018/05/05
//
/// @file    sdks_media.h
/// @brief   sdk接口
/// @author
/// @version 0.3
/// @warning 没有警告
///////////////////////////////////////////////////////////////////////////
#ifndef __SDKS_MEDIA_H__
#define __SDKS_MEDIA_H__
#include "sdk_def.h"

// multimedia  interface
#ifdef ANDROID
SDKS_API int sdks_init_android(void* javaVm);
SDKS_API void* sdks_md_get_wnd(void* p_view);
#endif
SDKS_API int sdks_md_live_start(unsigned int handle, int chn, int stream_type, void* p_wnd, bool is_hw_dec, SDK_PLAY_TIME_CB play_cb, void* p_obj);
SDKS_API int sdks_md_live_stop(unsigned int handle, int stream_id);
SDKS_API int sdks_md_chg_stream(unsigned int handle, int stream_id, int new_stream_type);

SDKS_API int sdks_md_pb_start(unsigned int handle, int chn, int stream_type, const char* s_time, void* p_wnd, bool is_hw_dec, SDK_PLAY_TIME_CB play_cb, void* p_obj);
SDKS_API int sdks_md_pb_seek(unsigned int handle, int stream_id, const char* time);
SDKS_API int sdks_md_pb_pause(unsigned int handle, int stream_id);
SDKS_API int sdks_md_pb_resume(unsigned int handle, int stream_id);
SDKS_API int sdks_md_pb_stop(unsigned int handle, int stream_id);
SDKS_API int sdks_md_set_pb_speed(unsigned int handle, int stream_id, int rate);

//record
SDKS_API int sdks_md_rec_start(unsigned int handle, int stream_id, const char* p_path);
SDKS_API int sdks_md_rec_start_width_time(unsigned int handle, int stream_id, const char* s_time, const char* e_time, const char* p_path);
SDKS_API int sdks_md_rec_stop(unsigned int handle, int stream_id);
SDKS_API int sdks_md_rec_percent(unsigned int handle, int stream_id);

//download
SDKS_API int  sdks_md_rec_download_start(unsigned int handle, int chn, int stream_type, const char* s_time, const char* e_time, const char* p_path);
SDKS_API int  sdks_md_rec_download_stop(unsigned int handle, int stream_id);

//capture
SDKS_API int sdks_md_capture(unsigned int handle, int stream_id, const char* p_path);

SDKS_API int sdks_md_audio_start(unsigned int handle, int stream_id);
SDKS_API int sdks_md_audio_stop(unsigned int handle, int stream_id);
SDKS_API int sdks_md_set_play_mode(unsigned int handle, int stream_id, int mode);


SDKS_API int sdks_md_talk_start(unsigned int handle, int chn, SDK_INTERCOM_DB_CB intercom_db_cb, void* obj);
SDKS_API int sdks_md_talk_stop(unsigned int handle, int chn);

#if defined(__RJ_WIN32__)
//fisheye and panoshow
SDKS_API int sdks_md_play_dynamic_video(unsigned int handle, int chn, int stream_type, int play_type, void* p_wnd, bool is_hw_dec, SDK_PLAY_TIME_CB play_cb, void* p_obj);
SDKS_API int sdks_md_set_auto_cruise(unsigned int handle, int stream_id, int is_auto_cruise);
SDKS_API int sdks_md_set_view_port(unsigned int handle, int stream_id, int left,int right,int top,int bottom);
SDKS_API int sdks_md_mouse_button_operate(unsigned int handle, int stream_id, int p_button, int p_action, int p_positionx, int p_positiony);
SDKS_API int sdks_md_mouse_drag_operate(unsigned int handle, int stream_id, int p_positionx, int p_positiony);
SDKS_API int sdks_md_mouse_scroll_operate(unsigned int handle, int stream_id, float p_delta, int p_positionx, int p_positiony);
SDKS_API int sdks_md_set_pano_show_mode(unsigned int handle, int stream_id, int p_show_mode);
SDKS_API int sdks_md_set_fisheye_mount_type(unsigned int handle, int stream_id, int p_mount_type);
SDKS_API int sdks_md_set_fisheye_ceiling_desktop_show_mode(unsigned int handle, int stream_id, int p_ceiling_desktop_show_mode);
SDKS_API int sdks_md_set_fisheye_wall_show_mode(unsigned int handle, int stream_id, int p_wall_show_mode);
SDKS_API int sdks_md_get_fisheye_rectify_parameters(unsigned int handle, int stream_id, char** p_result);
SDKS_API int sdks_md_set_fisheye_rectify_parameters(unsigned int handle, int stream_id, char* p_param);
#endif

#if defined(WIN32)
SDKS_API int  sdks_view_zoomin(unsigned int handle, int stream_id, int x, int y, int w, int h);///< 视频放大（选择区域）
SDKS_API int  sdks_view_zoominout_centern(unsigned int handle, int stream_id, int scale);///< 视频缩放（中心点）

#else
typedef enum {
	UIGestureRecognizerStateBegan_SDK = 1,
	UIGestureRecognizerStateChanged_SDK,
	UIGestureRecognizerStateEnd_SDK
} ScaleGesture;
///< scale： 缩放倍率   gesturestate  --->  ScaleGesture    w h显示区的宽高
SDKS_API int  sdks_view_zoom(int handle, int stream_id, float scale, int gesturestate, int w, int h); ///< 视频缩放
SDKS_API int  sdks_view_translate(int handle, int stream_id, float x, float y); ///< 视频移动
#endif

#endif
