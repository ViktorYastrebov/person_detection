#include "face_data.h"

#include "json/cJSON.h"

namespace sunell_camera {
    //TODO: add all checks for entries
    //      will fail for invalid json

    bool FaceDataVector::fromJsonData(const char *data) {
        bool ret = false;
        cJSON *root = cJSON_Parse(data);
        if(root)
        {
            cJSON *detectedList = cJSON_GetObjectItemEx(root, "TargetDetectList", cJSON_Array);
            if (detectedList) {
                ret = true;
                int length = cJSON_GetArraySize(detectedList);
                for (int i = 0; i < length; ++i) {
                    cJSON *entry = cJSON_GetArrayItem(detectedList, i);

                    cJSON*jsonValueX = cJSON_GetObjectItemEx(entry, "X", cJSON_Number);
                    cJSON*jsonValueY = cJSON_GetObjectItemEx(entry, "Y", cJSON_Number);
                    cJSON*jsonValueW = cJSON_GetObjectItemEx(entry, "W", cJSON_Number);
                    cJSON*jsonValueH = cJSON_GetObjectItemEx(entry, "H", cJSON_Number);

                    cJSON *personObj = cJSON_GetObjectItemEx(entry, "PersonFace", cJSON_Object);
                    cJSON *jsonConfidence = cJSON_GetObjectItemEx(personObj, "Confidence", cJSON_Number);
                    cJSON *jsonTemperature = cJSON_GetObjectItemEx(personObj, "Temperature", cJSON_Number);

                    FaceData face_data;
                    face_data.x = jsonValueX->valueint;
                    face_data.y = jsonValueY->valueint;
                    face_data.width = jsonValueW->valueint;
                    face_data.height = jsonValueH->valueint;
                    face_data.confidence = static_cast<float>(jsonConfidence->valuedouble);
                    face_data.temperature = jsonTemperature->valuedouble;
                    faces_data_.push_back(std::move(face_data));
                }
            }
            cJSON_Delete(root);
        }
        return ret;
    }
}