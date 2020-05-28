#pragma once 

#include "decl_spec.h"
#include "connection.h"
#include "stream_data_holder.h"
#include "face_data.h"

namespace ganz_camera {

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);
        void face_callback(unsigned int handle, int pic_type, void* p_data, int *data_len, void** p_result, void* p_obj);
    }

    class GANZ_CAMERA_ENGINE_DECL FaceHandler {
    public:
        FaceHandler(StreamDataHolder &holder, Connection &owner, const int channel, STREAM_TYPE type);
        ~FaceHandler();

        FaceHandler(const FaceHandler &) = delete;
        FaceHandler & operator = (const FaceHandler &) = delete;

        int getStreamId() const;

    private:
        friend void callback_wrapper::face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);
        friend void callback_wrapper::face_callback(unsigned int handle, int pic_type, void* p_data, int *data_len, void** p_result, void* p_obj);

        void handle(FaceDataVector &&faces);
        StreamDataHolder &holder_;
        Connection &owner_;
        int channel_;
        STREAM_TYPE stream_type_;
        int stream_id_;

        int start_face = -1;

    };
}