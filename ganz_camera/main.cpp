#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <windows.h>
#include "sdks.h"
//#include "sdk_face.h"
//#include "sdks_media.h"

#include "json/json.hpp"

#include <fstream>


void disconnect_handler(unsigned int handle, void *p_obj) {
    std::cout << "Disonnecting ..." << std::endl;
}

void alarm_handler(unsigned int handle, void** p_data, void* p_obj) {

    if (p_data != nullptr)
    {
        const char *data_ptr = static_cast<char*>(*p_data);
        std::string data(data_ptr);
        if (p_obj) {
            std::ofstream *ptr = static_cast<std::ofstream*>(p_obj);
            (*ptr) << "Alarm data :" << data << std::endl;
        }

        //std::cout << "Alarm data :" << data << std::endl;


        //解析报警信息
        //char p_result[1024] = { 0 };
        //strcpy(p_result, (char*)*p_data);
        //cJSON *p_json_root = cJSON_Parse((char*)*p_data);
        //cJSON* p_json_data = cJSON_GetObjectItemEx(p_json_root, "data", cJSON_Object);
        //cJSON* p_data_1 = cJSON_GetObjectItemEx(p_json_data, "dev_ip", cJSON_String);   //设备IP
        //cJSON* p_data_2 = cJSON_GetObjectItemEx(p_json_data, "src_type", cJSON_Number);  //报警源类型
        //cJSON* p_data_3 = cJSON_GetObjectItemEx(p_json_data, "src_id", cJSON_Number);    //报警源ID
        //cJSON* p_data_4 = cJSON_GetObjectItemEx(p_json_data, "dev_id", cJSON_String);     //设备ID
        //cJSON* p_data_5 = cJSON_GetObjectItemEx(p_json_data, "dev_type", cJSON_Number);   //设备类型
        //cJSON* p_data_6 = cJSON_GetObjectItemEx(p_json_data, "main_type", cJSON_Number);   //报警主类型
        //cJSON* p_data_7 = cJSON_GetObjectItemEx(p_json_data, "sub_type", cJSON_Number);    //报警次类型
        //cJSON* p_data_8 = cJSON_GetObjectItemEx(p_json_data, "alarm_flag", cJSON_Number);  //报警标志。1：开始，2：停止
        //cJSON* p_data_9 = cJSON_GetObjectItemEx(p_json_data, "time", cJSON_String);        //报警时间
        //cJSON* p_SNPointList = cJSON_GetObjectItemEx(p_json_root, "SNPointList", cJSON_Array);
        //for (int i = 0; i < cJSON_GetArraySize(p_SNPointList); i++)
        //{
        //    cJSON *item = cJSON_GetArrayItem(p_SNPointList, i);
        //    cJSON* p_SNPointList_1 = cJSON_GetObjectItemEx(item, "X", cJSON_Number);
        //    cJSON* p_SNPointList_2 = cJSON_GetObjectItemEx(item, "Y", cJSON_Number);
        //}
        //cJSON* p_AlarmAreaList = cJSON_GetObjectItemEx(p_json_root, "AlarmAreaList", cJSON_Array);
        //for (int i = 0; i < cJSON_GetArraySize(p_AlarmAreaList); i++)
        //{
        //    cJSON *item = cJSON_GetArrayItem(p_AlarmAreaList, i);
        //    cJSON* p_top = cJSON_GetObjectItemEx(item, "top", cJSON_Number);
        //    cJSON* p_bottom = cJSON_GetObjectItemEx(item, "bottom", cJSON_Number);
        //    cJSON* p_left = cJSON_GetObjectItemEx(item, "left", cJSON_Number);
        //    cJSON* p_right = cJSON_GetObjectItemEx(item, "right", cJSON_Number);
        //}
        //cJSON* p_json_Thermal = cJSON_GetObjectItemEx(p_json_root, "Thermal", cJSON_Object);
        //cJSON* p_MaxTemperature_X = cJSON_GetObjectItemEx(p_json_Thermal, "MaxTemperature_X", cJSON_Number);
        //cJSON* p_MaxTemperature_Y = cJSON_GetObjectItemEx(p_json_Thermal, "MaxTemperature_Y", cJSON_Number);
        //cJSON* p_MaxTemperature = cJSON_GetObjectItemEx(p_json_Thermal, "MaxTemperature", cJSON_Number);
        //cJSON* p_MinTemperature = cJSON_GetObjectItemEx(p_json_Thermal, "MinTemperature", cJSON_Number);
        //cJSON* p_TemperatureThreshold = cJSON_GetObjectItemEx(p_json_Thermal, "TemperatureThreshold", cJSON_Number);

        //cJSON_Delete(p_json_root);
    }
}

void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj) {
    if (p_data != nullptr) {
        const char *data_ptr = static_cast<char*>(*p_result);
        std::string data(data_ptr);

        if (p_obj) {
            std::ofstream *ptr = static_cast<std::ofstream*>(p_obj);
            (*ptr) << "face_detection_handler :" << data << std::endl;
        }

    }
    //if (p_pic_data != NULL)
    //{
    //    cJSON* p_root = cJSON_Parse((char*)*p_result);
    //    if (p_root == NULL)
    //    {
    //        return;
    //    }
    //    cJSON* pic_len = cJSON_GetObjectItemEx(p_root, "PictureLen", cJSON_Number);
    //    if (NULL == pic_len)
    //    {
    //        return;
    //    }
    //    char result[1024] = { 0 };
    //    sprintf(result, "%s", *p_result);
    //    FILE  *fp = fopen("E://picture//json.txt", "ab+");
    //    fprintf(fp, "%s\n", result);
    //    fflush(fp);
    //    fclose(fp);

    //    char* data = (char*)p_pic_data;
    //    char chTemp[64] = {};
    //    std::string pic_name = "E://picture//";
    //    sprintf(chTemp, "%s%d%s", "E://picture//test", npic_num, ".jpg");
    //    FILE  *file = fopen(chTemp, "wb+");
    //    if (file == NULL) {
    //        return;
    //    }
    //    size_t size = fwrite(data, 1, pic_len->valueint, file);
    //    fclose(file);
    //    npic_num++;
    //}
}


int main(int argc, char *argv[])
{
    std::string host;
    unsigned short port = 0;
    std::string user;
    std::string pwd;
    bool is_ssl = false;

    if (argc != 5) {
        std::cout << "Usage: server_demo.exe ssl/no-ssl IP user password";
        return 0;
    } else {
        std::string test_ssl(argv[1]);
        is_ssl = test_ssl == "ssl";
        host = std::string(argv[2]);
        // port = std::atoi(argv[3]);
        user = std::string(argv[3]);
        pwd = std::string(argv[4]);
    }

    std::ofstream ofs("output.txt");

    int ret = sdks_dev_init(nullptr);
    if(!ret) {
        //  http://79.26.226.220:9090
        // user :  admin 
        // password : K2K2020

        unsigned int handle = -1;
        if (is_ssl) {
            port = 20001;
            std::cout << "Going to connect via sdks_dev_conn_ssl with params :" << std::endl;
            std::cout << "Ip :" << host << ", port :" << port << std::endl;
            handle = sdks_dev_conn_ssl(host.c_str(),
                port,
                user.c_str(),
                pwd.c_str(),
                disconnect_handler,
                nullptr);
        } else {
            port = 30001;
            std::cout << "Going to connect via sdks_dev_conn with params :" << std::endl;
            std::cout << "Ip :" << host << ", port :" << port << std::endl;
            handle = sdks_dev_conn(host.c_str(),
                                port,
                                user.c_str(),
                                pwd.c_str(), 
                                disconnect_handler,
                                nullptr);
        }

        if (handle > 0) {
            std::cout << "Connected to camera by : " << host << ":" << port << std::endl;

            ret = sdks_dev_start_alarm(handle, alarm_handler, &ofs);
            std::cout << "Setup alarm :" << std::boolalpha << (ret == 0) << std::endl;

            int stream_id = sdks_dev_face_detect_start(handle, 1, 1, 4, face_detection_handler, &ofs);
            if (stream_id > 0) {
                std::cout << " Face detector setup" << std::endl;
            }

            std::string input;
            while (std::getline(std::cin, input )) {
                if (input == "exit") {
                    break;
                }
            }

            ret = sdks_dev_face_detect_stop(handle, stream_id);
            if (ret) {
                std::cout << "stopping face detection has failed" << std::endl;
            }

            ret = sdks_dev_stop_alarm(handle);
            if (ret) {
                std::cout << "stopping alarm notification has failed" << std::endl;
            }
            sdks_dev_conn_close(handle);
        } else {
            std::cout << "Can't connect to the camera by " << host << " IP" << ":" << port << std::endl;
        }
        sdks_dev_quit();
    }

    return 0;
}