#pragma once

#include <opencv2/core.hpp>

enum AVCodecID;
struct AVPacket;
struct AVCodec;
struct AVCodecParserContext;
struct AVCodecContext;
struct AVFrame;

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
            AVCodecParserContext *parser_;
            AVCodecContext *codec_ctx_;
            AVFrame *frame_;
        };
    }
}