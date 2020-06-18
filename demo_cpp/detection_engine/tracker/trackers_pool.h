#pragma once

#include "kalman_tracker.h"
#include <vector>


namespace tracker {

#pragma warning(push)
#pragma warning(disable: 4251)

    struct TRACKER_ENGINE TrackResult {
        cv::Rect bbox;
        int id;
    };

    class TRACKER_ENGINE TrackersPool {
    public:
        TrackersPool() = default;
        ~TrackersPool() = default;

        std::vector<TrackResult> update(const std::vector<cv::Rect> &detections);

    private:
        int counter_ = 0;
        bool initialized = false;
        int max_age_ = 1;
        int min_hits_ = 3;
        double iou_threshold_ = 0.3;
        std::vector<KalmanTracker> trackers_;

        int frame_counter_ = 0;
    };
#pragma warning(pop)
}