#pragma once

#include "tracker.h"
#include <vector>

namespace sort_tracker {

    class TrackerImpl {
    public:
        TrackerImpl(int max_age = 1, int min_hits = 3, int max_time_since_update = 32, int false_first_occurs_limit = 3);
        ~TrackerImpl() = default;

        std::vector< TrackResult > update(const common::datatypes::DetectionResults& detections);

       const std::vector<Track> &getTracks() const;

    private:
        common::datatypes::TrackerMatch process_match(const std::vector<common::datatypes::DetectionBox> &predicted, const common::datatypes::DetectionResults &detections);
    private:
        bool initialized_;
        int max_age_;
        int min_hits_;
        double iou_threshold_ = 0.3;
        std::vector<Track> trackers_;
        int frame_counter_ = 0;
        int max_time_since_update_;
        int first_false_occurance_limit_;

    };
}