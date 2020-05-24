#pragma once

#include <tbb/concurrent_queue.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <functional>

#include "camera_client/face_data.h"

namespace ganz_camera {

    class StreamDataHolder {
    public:
        using EntryProcessFunc = std::function<void(StreamDataHolder &holder, cv::Mat elem, const FaceDataVector& faces)>;

        StreamDataHolder();
        ~StreamDataHolder() = default;

        void start(EntryProcessFunc func);
        void stop();
        void put(cv::Mat frame);
        //INFO: current solution is:
        //      hold the queue of cv::Mat & queue of Faces
        //      on each iteraction pop cv::Mat and optional FaceVector if exists
        //      by default FacesVector is empty
        //      So it should give the sync between Picture & Faces data
        void put(FaceDataVector &&faces);

    private:
        std::atomic_bool stop_flag_;
        tbb::concurrent_queue<cv::Mat> data_;
        tbb::concurrent_queue<FaceDataVector> faces_;
    };
}