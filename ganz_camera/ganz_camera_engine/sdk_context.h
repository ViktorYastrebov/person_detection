#pragma once

#include "decl_spec.h"
#include <memory>
#include <list>
#include <atomic>

#pragma warning(push)
#pragma warning(disable: 4251)

namespace sunell_camera {


    namespace callback_wrapper {
        void disconnect_handler(unsigned int handle, void* p_obj);
    }

    class FaceDetector;

    enum PICTURE_SIZE : int {
        SMALL = 4,
        BIG = 5,
        SIZE_CHART = 7 // NEEDS TO CHECK
    };

    enum GANZ_CAMERA_ENGINE_DECL STREAM_TYPE : int {
        HD = 1,
        SD = 2
    };

    class GANZ_CAMERA_ENGINE_DECL SDKContext final {
    public:
        SDKContext();
        ~SDKContext();
        std::shared_ptr<FaceDetector> createFaceDetector(const std::string &host, const std::string &user, const std::string &pwd, bool ssl, const int channel, STREAM_TYPE type, PICTURE_SIZE size);
    private:
        friend void callback_wrapper::disconnect_handler(unsigned int handle, void* p_obj);
        static std::atomic<int> counter_;
    };
}

#pragma warning(pop)