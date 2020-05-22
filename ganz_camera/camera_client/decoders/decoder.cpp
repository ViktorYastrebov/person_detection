#include "decoder.h"
#include <exception>

namespace ganz_camera {
    namespace decoders {

        H264Decoder::H264Decoder(AVCodecID codec_id)
            :codec_id_(codec_id)
        {
            packet_ = av_packet_alloc();
            if (!packet_) {
                throw std::exception("av_packet_alloc has failed");
            }
            codec_ = avcodec_find_decoder(codec_id_);
            if (!codec_) {
                throw std::exception("Count not create codec by avcodec_find_decoder");
            }
            parser_ = av_parser_init(codec_->id);
            if (!parser_) {
                throw std::exception("av_parser_init has failed");
            }
            codec_ctx_ = avcodec_alloc_context3(codec_);
            if (!codec_ctx_) {
                throw std::exception("avcodec_alloc_context3 has failed");
            }
            if (avcodec_open2(codec_ctx_, codec_, nullptr)) {
                throw std::exception("avcodec_open2 has failed");
            }
            frame_ = av_frame_alloc();
        }

        H264Decoder::~H264Decoder()
        {

        }

        void H264Decoder::decode(const unsigned char *data, int data_length)
        {
            std::size_t data_size = data_length;
            while (data_size > 0) {
                int ret = av_parser_parse2(parser_,
                                       codec_ctx_,
                                       &packet_->data,
                                       &packet_->size,
                                       data,
                                       data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
                if (ret < 0) {
                    //fprintf(stderr, "Error while parsing\n");
                    //exit(1);
                }
                data += ret;
                data_size -= ret;

                if (packet_->size) {
                   //decode(c, frame, pkt, outfilename);
                }
            }
        }

        AVPacket* packet_;
        AVCodec *codec_;
        AVCodecParserContext *parser_;
        AVCodecContext *codec_ctx_;
    }
}