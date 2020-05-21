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