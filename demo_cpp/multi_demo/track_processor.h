#pragma once

#include "display_window.h"
#include "trackers_pool.h"
#include <thread>
#include <atomic>
#include <tbb/concurrent_queue.h>
#include <base_model.h>


class TrackProcessor {
public:

    struct TrackInputData {
        cv::Mat frame;
        std::vector< DetectionResult> detections;
    };

    TrackProcessor(const int idx, CentralWidget &cw);
    ~TrackProcessor();

    void start();
    void stop();
    void put(TrackInputData &&input);

    int getId() const;

protected:
    void processingImpl();
private:
    int idx_;
    CentralWidget &central_;
    std::atomic_bool stop_flag_;
    tracker::TrackersPool tracks_;
    tbb::concurrent_queue<TrackInputData> detections_;
    std::thread runner_;
};