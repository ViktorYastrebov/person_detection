#pragma once

#include "kalman_tracker.h"
#include <vector>
#include "datatypes.h"

namespace sort_tracker {

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

        //std::vector<TrackResult> update(const std::vector<DetectionResult> &detections);
       std::vector<TrackResult>  update(const common::datatypes::DetectionResults &detections);
       //void update(const common::datatypes::DetectionResults& detections);

       const std::vector<SortTracker> &getTracks() const;

    private:
        common::datatypes::TrackerMatch process_match(const std::vector<common::datatypes::DetectionBox> &predicted, const common::datatypes::DetectionResults &detections);
    private:
        bool initialized_;
        int max_age_;
        int min_hits_;
        double iou_threshold_ = 0.3;
        std::vector<SortTracker> trackers_;

        int frame_counter_ = 0;
    };
#pragma warning(pop)
}