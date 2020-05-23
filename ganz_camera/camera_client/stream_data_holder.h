#pragma once

#include <tbb/concurrent_queue.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <functional>

namespace ganz_camera {

    class StreamDataHolder {
    public:
        StreamDataHolder();
        ~StreamDataHolder() = default;

        void start(std::function<void(StreamDataHolder &holder, cv::Mat elem)> func);
        void stop();
        void put(cv::Mat frame);

    private:
        std::atomic_bool stop_flag_;
        tbb::concurrent_queue<cv::Mat> data_;
    };
}