#include "client.h"

#include "sdks.h"

namespace wrapper {
    void disconnect_handler(unsigned int handle, void *p_obj) {
        //std::cout << "Disonnecting ..." << std::endl;
    }

   void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj) {

       ganz_camera::client::CameraClient* owner = static_cast<ganz_camera::client::CameraClient*>(p_obj);
       owner->face_handle_callback();
        //if (p_data != nullptr) {
        //    const char *data_ptr = static_cast<char*>(*p_result);
        //    std::string data(data_ptr);

        //    if (p_obj) {
        //        std::ofstream *ptr = static_cast<std::ofstream*>(p_obj);
        //        (*ptr) << "face_detection_handler :" << data << std::endl;
        //    }

        //}
    }

   void alarm_handler(unsigned int handle, void** p_data, void* p_obj) {
       ganz_camera::client::CameraClient* owner = static_cast<ganz_camera::client::CameraClient*>(p_obj);
       if (p_data != nullptr)
       {
           const char *data_ptr = static_cast<char*>(*p_data);
           std::string data(data_ptr);
           //owner->alarm_handle_callback();
       }
   }

}


namespace ganz_camera {
    namespace client {
        CameraClient::CameraClient(const std::string &host, const std::string &user, const std::string &pwd, bool ssl)
            : init_ctx_()
            , host_(host)
            , user_(user)
            , pwd_(pwd)
            , port_(ssl ? 20001 : 30001)
            , conn_handle_(0) // invalid by default
            , stream_id_(-1)
        {
            if (ssl) {
                conn_handle_ = sdks_dev_conn_ssl(host_.c_str(),
                                                 port_,
                                                 user_.c_str(),
                                                 pwd_.c_str(),
                                                 wrapper::disconnect_handler,
                                                 nullptr);
            }
            else {
                conn_handle_ = sdks_dev_conn(host_.c_str(),
                                            port_,
                                            user_.c_str(),
                                            pwd_.c_str(),
                                            wrapper::disconnect_handler,
                                            nullptr);
            }

            if (!conn_handle_) {
                std::string error = "Can't connect to server : " + host + ":"+ std::to_string(port_);
                throw std::exception(error.c_str());
            }
        }
        
        CameraClient::~CameraClient()
        {
            if(conn_handle_ ) {
                sdks_dev_conn_close(conn_handle_);
            }
        }

        void CameraClient::face_handle_callback() {}
        void CameraClient::alarm_handle_callback() {}

        void CameraClient::start_face_detection() {
            //TODO: add picture type 4 = small, 5= big, 7 = size chart
            stream_id_ = sdks_dev_face_detect_start(conn_handle_, 1, 1, 5, wrapper::face_detection_handler, this);
            if (stream_id_ <= 0) {
                throw std::exception("sdks_dev_face_detect_start has failed");
            }
        }

        void CameraClient::stop_face_detection() {
            if (stream_id_ > 0) {
                sdks_dev_face_detect_stop(conn_handle_, stream_id_);
            }
        }

        void CameraClient::start_alarm_handling() {
            int ret = sdks_dev_start_alarm(conn_handle_, wrapper::alarm_handler, this);
        }

        void CameraClient::stop_alarm_handling() {
            int ret = sdks_dev_stop_alarm(conn_handle_);
        }

    }
}