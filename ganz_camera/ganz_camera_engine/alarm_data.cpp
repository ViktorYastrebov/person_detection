#include "alarm_data.h"

#include "json/cJSON.h"

namespace ganz_camera {

    bool AlarmData::fromJsonData(const char *data) {
        bool ret = false;
        cJSON *root_node = cJSON_Parse(data);
        if (root_node) {
            ret = true;

            cJSON* p_time_str = cJSON_GetObjectItemEx(root_node, "time", cJSON_String);
            time.assign(static_cast<const char*>(p_time_str->string));

            cJSON* p_SNPointList = cJSON_GetObjectItemEx(root_node, "SNPointList", cJSON_Array);
            if (p_SNPointList) {
                int length = cJSON_GetArraySize(p_SNPointList);
                for (int i = 0; i < length; ++i) {
                    cJSON *item = cJSON_GetArrayItem(p_SNPointList, i);
                    cJSON* jsonX = cJSON_GetObjectItemEx(item, "X", cJSON_Number);
                    cJSON* jsonY = cJSON_GetObjectItemEx(item, "Y", cJSON_Number);
                    sn_points.push_back(cv::Point2i(jsonX->valueint, jsonY->valueint));
                }

                cJSON* p_AlarmAreaList = cJSON_GetObjectItemEx(root_node, "AlarmAreaList", cJSON_Array);
                length = cJSON_GetArraySize(p_AlarmAreaList);
                for (int i = 0; i < length; ++i)
                {
                    cJSON *item = cJSON_GetArrayItem(p_AlarmAreaList, i);
                    cJSON* p_top = cJSON_GetObjectItemEx(item, "top", cJSON_Number);
                    cJSON* p_bottom = cJSON_GetObjectItemEx(item, "bottom", cJSON_Number);
                    cJSON* p_left = cJSON_GetObjectItemEx(item, "left", cJSON_Number);
                    cJSON* p_right = cJSON_GetObjectItemEx(item, "right", cJSON_Number);
                    
                    //INFO: keep in mind that the cv::Rect args are top, bottom left, right
                    alarm_areas.push_back(
                        cv::Rect(p_left->valueint, p_top->valueint, p_right->valueint, p_bottom->valueint)
                    );
                }
                cJSON_Delete(root_node);
            }
        }
        return ret;
    }
}