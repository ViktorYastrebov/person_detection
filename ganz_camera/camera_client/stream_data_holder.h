#pragma once

#include <tbb/concurrent_queue.h>
#include "camera_client/base_video_stream.h"

#include <atomic>
#include <functional>

#include "camera_client/face_data.h"

namespace ganz_camera {

    class StreamDataHolder {
    public:
        using EntryProcessFunc = std::function<void(StreamDataHolder &holder, const FrameInfo &info, const FaceDataVector& faces)>;

        StreamDataHolder();
        ~StreamDataHolder();

        void start(EntryProcessFunc func);
        void stop();
        void put(FrameInfo &&info);
        //INFO: current solution is:
        //      hold the queue of cv::Mat & queue of Faces
        //      on each iteraction pop cv::Mat and optional FaceVector if exists
        //      by default FacesVector is empty
        //      So it should give the sync between Picture & Faces data
        void put(FaceDataVector &&faces);
    protected:
        void process();
    private:
        std::atomic_bool stop_flag_;
        tbb::concurrent_queue<FrameInfo> data_;
        tbb::concurrent_queue<FaceDataVector> faces_;
        EntryProcessFunc processor_;
        std::thread runner_;
    };
}