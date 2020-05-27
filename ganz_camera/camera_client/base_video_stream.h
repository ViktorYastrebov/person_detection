#pragma once

#include "camera_client/connection.h"
#include <opencv2/core.hpp>

namespace ganz_camera {

    class StreamDataHolder;

    struct FrameInfo {
        cv::Mat frame;
        long long nAbsoluteTimeStamp;
        long long nRelativeTimeStamp;
    };

    class BaseVideoStream {
    public:
        BaseVideoStream(StreamDataHolder &holder);
        virtual ~BaseVideoStream() = default;

        virtual void Start() = 0;
        virtual void Stop() = 0;

    protected:
        void handleFrame(FrameInfo &&frameInfo);
    private:
        StreamDataHolder &data_holder_;
    };
}