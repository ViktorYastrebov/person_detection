#pragma once 

#include "connection.h"
#include "camera_client/stream_data_holder.h"

#include "camera_client/face_data.h"

//INFO: it does not work with nlohmann::json library due to encoding problem
//      Currently it supports utf-8 which is not default windows encoding
//      
//      Moved to cJson library for testing
//      
//#include "json/json.hpp"

namespace ganz_camera {

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);
    }

    class FaceHandler {
    public:
        FaceHandler(StreamDataHolder &holder, Connection &owner, const int channel, STREAM_TYPE type);
        ~FaceHandler();

        FaceHandler(const FaceHandler &) = delete;
        FaceHandler & operator = (const FaceHandler &) = delete;

        int getStreamId() const;

    private:
        friend void callback_wrapper::face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);

        void handle(FaceDataVector &&faces);
            //void handle(const nlohmann::json &json, const char *picture);
        StreamDataHolder &holder_;
        Connection &owner_;
        int channel_;
        STREAM_TYPE stream_type_;
        int stream_id_;
    };
}