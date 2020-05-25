#include "camera_client/video_stream.h"
#include "sdks.h"

#include <thread>
#include <chrono>

#include <iostream>

namespace ganz_camera {

    namespace callback_wrapper {

        //std::string to_encoding(int code) {
        //    static const std::unordered_map<int, std::string> formats = {
        //        {0, "MPEG4"},
        //        {1, "H264"},
        //        {2,"MJPEG"},
        //        {3, "SVC"},
        //        {6, "JPEG"},
        //        {7, "H265(base)"},
        //        {8, "H265(main)"},
        //        {9, "H265(high)"},
        //        {101, "G7231"},
        //        {102, "G711A"},
        //        {103, "G711U"},
        //        {104, "G722"},
        //        {105, "G726"},
        //        {106, "G729"},
        //        {107, "AMR"},
        //        {108, "PCM"}
        //    };
        //    std::unordered_map<int, std::string>::const_iterator it = formats.find(code);
        //    if (it != formats.cend()) {
        //        return it->second;
        //    }
        //    return "Unknown";
        //}

        void stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj)
        {
            VideoStream *owner = static_cast<VideoStream*>(p_obj);
            if (p_data != NULL) {
                ST_AVFrameData *p_frame = (ST_AVFrameData*)p_data;
                 unsigned char *data_ptr = (unsigned char*)p_frame->pszData;
                 owner->handle(data_ptr, p_frame->nDataLength);
            }
        }
    }

    VideoStream::VideoStream(StreamDataHolder &holder, Connection &owner, const int channel, STREAM_TYPE type)
        :BaseVideoStream(holder)
        , owner_(owner)
        , stream_type_(type)
        , channel_(channel)
        , h264_decoder_()
    {}

    void VideoStream::Start() {
        stream_id_ = sdks_dev_live_start(owner_.getHandle(), channel_, stream_type_, callback_wrapper::stream_handler, this);
        if (stream_id_ < 0) {
            throw std::exception("sdks_dev_live_start has failed");
        }
    }

    void VideoStream::Stop() {
        sdks_dev_live_stop(owner_.getHandle(), stream_id_);
    }

    VideoStream::~VideoStream()
    {}

    void VideoStream::handle(unsigned char *data, int data_length) {
        cv::Mat frame = h264_decoder_.decode(data, data_length);
        if (!frame.empty()) {
            //std::cout << "Has valid mat data";
            //INFO: it works fine without it. tested !!!
            BaseVideoStream::handleFrame(frame);
        }
    }
}