#include "decoder.h"
#include <exception>

extern "C" {
#include "libswscale/swscale.h"
}

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
            //parser_ = av_parser_init(codec_->id);
            //if (!parser_) {
            //    throw std::exception("av_parser_init has failed");
            //}
            codec_ctx_ = avcodec_alloc_context3(codec_);
            if (!codec_ctx_) {
                throw std::exception("avcodec_alloc_context3 has failed");
            }
            if (avcodec_open2(codec_ctx_, codec_, nullptr)) {
                throw std::exception("avcodec_open2 has failed");
            }
            frame_ = av_frame_alloc();
            //BGR_frame_ = av_frame_alloc();
        }

        H264Decoder::~H264Decoder()
        {
            //av_frame_free(&BGR_frame_);
            av_frame_free(&frame_);
            avcodec_free_context(&codec_ctx_);
            av_packet_free(&packet_);
        }

        cv::Mat H264Decoder::decode(unsigned char *data, int data_length)
        {
            if (!data_length) {
                return cv::Mat();
            }

            packet_->data = data;
            packet_->size = data_length;

            int got_frame = 0;
            int processed_len = avcodec_decode_video2(codec_ctx_, frame_, &got_frame, packet_);
            if (processed_len < 0) {
                return cv::Mat();
            }

            int BGRsize = avpicture_get_size(AV_PIX_FMT_BGR24, codec_ctx_->width, codec_ctx_->height);
            uint8_t *out_buffer = (uint8_t *)av_malloc(BGRsize);

            AVFrame *BGR_frame_ = av_frame_alloc();
            avpicture_fill((AVPicture *)BGR_frame_, out_buffer, AV_PIX_FMT_BGR24, codec_ctx_->width, codec_ctx_->height);

            struct SwsContext *img_convert_ctx = sws_getContext(codec_ctx_->width,
                                                                codec_ctx_->height,
                                                                codec_ctx_->pix_fmt,
                                                                codec_ctx_->width,
                                                                codec_ctx_->height,
                                                                AV_PIX_FMT_BGR24,
                                                                SWS_BICUBIC, NULL, NULL, NULL);
            //pCvMat.create(cv::Size(codec_ctx_->width, codec_ctx_->height), CV_8UC3);
            
            if (got_frame) {
                sws_scale(img_convert_ctx,
                         (const uint8_t *const *)frame_->data,
                         frame_->linesize, 0,
                         codec_ctx_->height,
                         BGR_frame_->data,
                         BGR_frame_->linesize);

                cv::Mat ret(cv::Size(codec_ctx_->width, codec_ctx_->height), CV_8UC3);
                std::memcpy(ret.data, out_buffer, BGRsize);
                av_free(out_buffer);
                av_frame_free(&BGR_frame_);
                return ret;
            }
            av_frame_free(&BGR_frame_);
            av_free(out_buffer);
            return cv::Mat();
        }
    }
}