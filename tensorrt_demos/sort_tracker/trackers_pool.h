#pragma once

#include "kalman_tracker.h"
#include "common/datatypes.h"
#include <vector>

namespace tracker {

#pragma warning(push)
#pragma warning(disable: 4251)

    struct TRACKER_ENGINE TrackResult {
        cv::Rect bbox;
        int id;
        int class_id;
    };

    class TRACKER_ENGINE TrackersPool {
    public:
        TrackersPool(int max_age = 1, int min_hits = 3);
        ~TrackersPool() = default;

        std::vector<TrackResult> update(const common::datatypes::DetectionResults &detections);

    private:
        bool initialized_;
        int max_age_;
        int min_hits_;
        double iou_threshold_ = 0.3;
        std::vector<KalmanTracker> trackers_;

        int frame_counter_ = 0;
    };
#pragma warning(pop)
}