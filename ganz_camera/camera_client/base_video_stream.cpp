#include "base_video_stream.h"
#include "camera_client/stream_data_holder.h"

namespace ganz_camera {

    BaseVideoStream::BaseVideoStream(StreamDataHolder &holder)
        :data_holder_(holder)
    {}

    void BaseVideoStream::handleFrame(FrameInfo &&frameInfo)
    {
        data_holder_.put(std::move(frameInfo));
    }
}