#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <opencv2/core.hpp>


namespace ganz_camera {
    namespace decoders {

        class H264Decoder final {
        public:
            H264Decoder();
            ~H264Decoder();
            cv::Mat decode(unsigned char *data, int data_length);
        private:
            AVCodecID codec_id_;
            AVPacket* packet_;
            AVCodec *codec_;
            AVCodecContext *codec_ctx_;
            AVFrame *frame_;
            //AVFrame *BGR_frame_;
        };
    }
}