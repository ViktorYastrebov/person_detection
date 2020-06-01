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

    std::atomic<int> SDKContext::counter_ = 0;

    SDKContext::SDKContext()
    {
        if (!counter_) {
            int ret = sdks_dev_init(nullptr);
            if (ret) {
                throw std::exception("SDK initialization error occurs");
            }
            ++counter_;
        }
    }

    SDKContext::~SDKContext() {
        --counter_;
        if (!counter_) {
            sdks_dev_quit();
        }
    }

    std::shared_ptr<FaceDetector> SDKContext::createFaceDetector(const std::string &host, const std::string &user, const std::string &pwd, bool ssl,
                                                                 const int channel, STREAM_TYPE type, PICTURE_SIZE size)
    {
        unsigned int connection = 0;
        unsigned short port = ssl ? 20001 : 30001;
        if (ssl) {
            connection = sdks_dev_conn_ssl(host.c_str(),
                port,
                user.c_str(),
                pwd.c_str(),
                callback_wrapper::disconnect_handler,
                this);
        } else {
            connection = sdks_dev_conn(host.c_str(),
                port,
                user.c_str(),
                pwd.c_str(),
                callback_wrapper::disconnect_handler,
                this);
        }

        if (!connection) {
            std::string error = "Can't connect to server : " + host + ":" + std::to_string(port);
            throw std::exception(error.c_str());
        }

        return std::shared_ptr<FaceDetector>(new FaceDetector(connection, channel, type, size));
    }

}