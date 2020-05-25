#include "decoder.h"
#include <exception>

extern "C" {
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
}

#include <iostream>

namespace ganz_camera {
    namespace decoders {

        H264Decoder::H264Decoder()
        {
            packet_ = av_packet_alloc();
            if (!packet_) {
                throw std::exception("av_packet_alloc has failed");
            }
            codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
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
            av_frame_free(&frame_);
            avcodec_free_context(&codec_ctx_);
            av_packet_free(&packet_);
        }

        cv::Mat H264Decoder::decode(unsigned char *data, int data_length)
        {
            if (!data_length) {
                return cv::Mat();
            }
            unsigned char *data_ptr = new unsigned char[data_length + AV_INPUT_BUFFER_PADDING_SIZE];
            std::memcpy(data_ptr, data, data_length);
            std::memset(data_ptr + data_length, 0, AV_INPUT_BUFFER_PADDING_SIZE);

            packet_->size = data_length;
            packet_->data = data_ptr;

            int has_frame = 0;
            avcodec_decode_video2(codec_ctx_, frame_, &has_frame, packet_);
            if (has_frame) {
                cv::Mat convert_mat(codec_ctx_->width, codec_ctx_->height, CV_8UC3);
                int cvLinesizes[1];
                cvLinesizes[0] = convert_mat.step1();

                SwsContext* conversion =
                    sws_getContext(codec_ctx_->width,
                        codec_ctx_->height,
                        (AVPixelFormat)frame_->format,
                        codec_ctx_->width,
                        codec_ctx_->height,
                        AV_PIX_FMT_BGR24,
                        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
                sws_scale(conversion, frame_->data, frame_->linesize, 0, codec_ctx_->height, &convert_mat.data, cvLinesizes);
                sws_freeContext(conversion);
                delete[] data_ptr;
                return convert_mat;
            }
            delete[] data_ptr;
            return cv::Mat();

#if 0
            int data_len = data_length;
            while (data_len) {
                int len = av_parser_parse2(parser_,
                    codec_ctx_,
                    &packet_->data,
                    &packet_->size,
                    data_ptr, data_length, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0
                );
                data_ptr += len;
                data_len -= len;
            }

            //https://timvanoosterhout.wordpress.com/2015/07/02/converting-an-ffmpeg-avframe-to-and-opencv-mat/
            if (packet_->size) {
                int ret = avcodec_send_packet(codec_ctx_, packet_);
                if (ret == 0) {
                    ret = avcodec_receive_frame(codec_ctx_, frame_);
                    if (ret) {
                        delete[] data_ptr;
                        return cv::Mat();
                    }

                    cv::Mat convert_mat(codec_ctx_->width, codec_ctx_->height, CV_8UC3);
                    int cvLinesizes[1];
                    cvLinesizes[0] = convert_mat.step1();

                    SwsContext* conversion =
                        sws_getContext(codec_ctx_->width,
                            codec_ctx_->height,
                            (AVPixelFormat)frame_->format,
                            codec_ctx_->width,
                            codec_ctx_->height,
                            AV_PIX_FMT_BGR24,
                            SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
                    sws_scale(conversion, frame_->data, frame_->linesize, 0, codec_ctx_->height, &convert_mat.data, cvLinesizes);
                    sws_freeContext(conversion);
                    delete[] data_ptr;
                    return convert_mat;
                }
            }
            delete[] data_ptr;
            return cv::Mat();
#endif
        }
    }
}