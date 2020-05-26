#pragma once

#include "camera_client/base_video_stream.h"
#include "camera_client/connection.h"
#include "camera_client/stream_data_holder.h"
#include "decoders/decoder.h"

#include <atomic>

namespace ganz_camera {

    // https://stackoverflow.com/questions/29263090/ffmpeg-avframe-to-opencv-mat-conversion

    namespace callback_wrapper {
        void stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj);
    }

    class /*[[deprecated("It does not work due to Decode problem(ffmpeg with undefined behaviour)")]]*/
    VideoStream : public BaseVideoStream {
    public:
        VideoStream(StreamDataHolder &holder, Connection &owner, const int channel, STREAM_TYPE type);

        virtual void Start() override;
        virtual void Stop() override;

        virtual ~VideoStream();
    private:
        friend void callback_wrapper::stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj);
        void handle(unsigned char *data, int data_length);
    private:
        Connection &owner_;
        STREAM_TYPE stream_type_;
        int channel_;
        int stream_id_;
        decoders::H264Decoder h264_decoder_;
        //std::atomic_bool stop_flag_;
    };

}