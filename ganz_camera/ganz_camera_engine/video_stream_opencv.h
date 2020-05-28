#pragma once

#include "camera_client/base_video_stream.h"
#include <opencv2/videoio.hpp>
#include <string>
#include <atomic>
#include <thread>

namespace ganz_camera {

    class SimpleVideoStream : public BaseVideoStream {
    public:
        SimpleVideoStream(StreamDataHolder &holder, const std::string &url);
        virtual ~SimpleVideoStream() = default;

        virtual void Start() override;
        virtual void Stop() override;
    private:
        void process();
    private:
        std::string url_;
        cv::VideoCapture stream_;
        std::atomic_bool stop_flag_;
        std::thread runner_;
    };

}