#pragma once

#include "camera_client/connection.h"

namespace ganz_camera {

    // https://stackoverflow.com/questions/29263090/ffmpeg-avframe-to-opencv-mat-conversion

    namespace callback_wrapper {
        void stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj);
    }

    class VideoStream {
    public:
        enum STREAM_TYPE: int {
            HD = 1,
            SD = 2
            //NOT_SUPPORTED_SMOOTH = 3
        };

        VideoStream(Connection &owner, const int channel, STREAM_TYPE type);
        ~VideoStream();
    private:
        friend void callback_wrapper::stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj);

        //void handle();

        Connection &owner_;
        STREAM_TYPE stream_type_;
        int channel_;
        int stream_id_;
    };

}