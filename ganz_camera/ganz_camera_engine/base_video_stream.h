#pragma once

#include "connection.h"
#include <opencv2/core.hpp>

#pragma warning(push)
#pragma warning(disable: 4251)

namespace ganz_camera {

    class GANZ_CAMERA_ENGINE_DECL StreamDataHolder;

    struct GANZ_CAMERA_ENGINE_DECL FrameInfo {
        cv::Mat frame;
        long long nAbsoluteTimeStamp;
        long long nRelativeTimeStamp;
    };

    class GANZ_CAMERA_ENGINE_DECL BaseVideoStream {
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

#pragma warning(pop)
