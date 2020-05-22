#pragma once

#include <libavcodec/avcodec.h>

namespace ganz_camera {
    namespace decoders {

        class H264Decoder final {
        public:
            H264Decoder(AVCodecID codec_id);
            ~H264Decoder();
            void decode(const unsigned char *data, int data_length);
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