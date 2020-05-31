#include "sdk_context.h"
#include "sdks.h"
#include "face_detector.h"

#include <string>
#include <exception>

namespace sunell_camera {

    namespace callback_wrapper {
        void disconnect_handler(unsigned int handle, void* p_obj) {
            // DO NOTHING
        }
    }

    SDKContext::SDKContext(const std::string &host, const std::string &user, const std::string &pwd, bool ssl)
        : host_(host)
        , user_(user)
        , pwd_(pwd)
        , port_(ssl ? 20001 : 30001)
        , is_ssl_(ssl)
    {
        int ret = sdks_dev_init(nullptr);
        if (ret) {
            throw std::exception("SDK initialization error occurs");
        }
    }

    SDKContext::~SDKContext() {
        sdks_dev_quit();
    }

    std::shared_ptr<FaceDetector> SDKContext::createFaceDetector(const int channel, STREAM_TYPE type, PICTURE_SIZE size) {
        unsigned int conn = 0;
        if (is_ssl_) {
            conn = sdks_dev_conn_ssl(host_.c_str(),
                port_,
                user_.c_str(),
                pwd_.c_str(),
                callback_wrapper::disconnect_handler,
                this);
        } else {
            conn = sdks_dev_conn(host_.c_str(),
                port_,
                user_.c_str(),
                pwd_.c_str(),
                callback_wrapper::disconnect_handler,
                this);
        }

        if (!conn) {
            std::string error = "Can't connect to server : " + host_ + ":" + std::to_string(port_);
            throw std::exception(error.c_str());
        }
        return std::shared_ptr<FaceDetector>(new FaceDetector(conn, channel, type, size));
    }

}