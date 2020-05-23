#pragma once

#include "camera_client/connection.h"
#include "camera_client/stream_data_holder.h"
#include <opencv2/core.hpp>

namespace ganz_camera {

    class BaseVideoStream {
    public:
        BaseVideoStream(StreamDataHolder &holder);
        virtual ~BaseVideoStream() = default;

        virtual void Start() = 0;
        virtual void Stop() = 0;

    protected:
        void handleFrame(cv::Mat frame);
    private:
        StreamDataHolder &data_holder_;
    };
}