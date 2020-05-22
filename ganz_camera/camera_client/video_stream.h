#pragma once

#include "camera_client/connection.h"
#include "decoders/decoder.h"

namespace ganz_camera {

    // https://stackoverflow.com/questions/29263090/ffmpeg-avframe-to-opencv-mat-conversion

    namespace callback_wrapper {
        void stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj);
    }

    //INFO: it hangs after passing some frames,
    //      It relates to Decode part. If it's turned off it works fine
    class VideoStream {
    public:
        enum STREAM_TYPE: int {
            HD = 1,
            SD = 2
            //NOT_SUPPORTED_SMOOTH = 3
        };

        //INFO: cv::Mat is intrusive ptr so its safe to pass by value
        using OutHandler = std::function<void(cv::Mat image)>;

        VideoStream(Connection &owner, const int channel, STREAM_TYPE type, OutHandler handler);
        ~VideoStream();
    private:
        friend void callback_wrapper::stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj);

        void handle(unsigned char *data, int data_length);

        Connection &owner_;
        STREAM_TYPE stream_type_;
        int channel_;
        int stream_id_;
        decoders::H264Decoder h264_decoder_;
        OutHandler handler_;
    };

}