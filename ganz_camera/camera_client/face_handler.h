#pragma once 

#include "connection.h"

#include "json/json.hpp"

namespace ganz_camera {

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);
    }

    class FaceHandler {
    public:
        FaceHandler(Connection &owner);
        ~FaceHandler();

        FaceHandler(const FaceHandler &) = delete;
        FaceHandler & operator = (const FaceHandler &) = delete;

        int getStreamId() const;

    private:
        friend void callback_wrapper::face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);

        void handle(const nlohmann::json &json, const char *picture);

        Connection &owner_;
        int stream_id_;
    };
}