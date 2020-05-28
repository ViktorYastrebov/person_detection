#pragma once

#include "base_video_stream.h"
#include "connection.h"
#include "stream_data_holder.h"
#include "decoders/decoder.h"

#include "sdks.h"

#pragma warning(push)
#pragma warning(disable: 4251)

namespace ganz_camera {

    namespace callback_wrapper {
        void stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj);
    }

    class GANZ_CAMERA_ENGINE_DECL VideoStream : public BaseVideoStream {
    public:
        VideoStream(StreamDataHolder &holder, Connection &owner, const int channel, STREAM_TYPE type);

        virtual void Start() override;
        virtual void Stop() override;

        virtual ~VideoStream();
    private:
        friend void callback_wrapper::stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj);
        void handle(const ST_AVFrameData *);
    private:
        Connection &owner_;
        STREAM_TYPE stream_type_;
        int channel_;
        int stream_id_;
        decoders::H264Decoder h264_decoder_;
    };
}

#pragma warning(pop)