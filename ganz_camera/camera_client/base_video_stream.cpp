#include "base_video_stream.h"

namespace ganz_camera {

    BaseVideoStream::BaseVideoStream(StreamDataHolder &holder)
        :data_holder_(holder)
    {}

    void BaseVideoStream::handleFrame(cv::Mat frame)
    {
        data_holder_.put(frame);
    }

}