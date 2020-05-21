#pragma once

#include <string>
#include <list>

#include "sdk_context.h"
#include "face_data_handler.h"
#include "alarm_data_handler.h"

namespace ganz_camera {
    namespace client {

        class CameraClient final {
        public:
            CameraClient(const std::string &host, const std::string &user, const std::string &pwd, bool ssl = false);
            ~CameraClient();

            CameraClient(const CameraClient &) = delete;
            CameraClient & operator = (const CameraClient &) = delete;

            void face_handle_callback();
            void alarm_handle_callback();

            void start_face_detection();
            void stop_face_detection();

            void start_alarm_handling();
            void stop_alarm_handling();

            //void attach(ganz_camera::FaceDataHandler *handler);
            //void attach(ganz_camera::AlarmHandler *handler);

        private:
            SDKContext init_ctx_;
            std::string host_;
            std::string user_;
            std::string pwd_;
            unsigned short port_;
            unsigned int conn_handle_;

            int stream_id_;
        };

    }
}