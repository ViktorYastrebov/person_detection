#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "common/datatypes.h"
#include "decl_spec.h"
#include "track.h"

namespace sort_tracker {

#pragma warning(push)
#pragma warning(disable: 4251)
    struct TRACKER_ENGINE TrackResult {
        cv::Rect bbox;
        int id;
        int class_id;
        float vx;
        float vy;
    };
#pragma warning(pop)

    class TrackerImpl;

    class TRACKER_ENGINE Tracker {
    public:
        Tracker(int max_age = 1, int min_hits = 3, int max_time_since_update = 32, int false_first_occurs_limit = 3);
        ~Tracker();
        std::vector< TrackResult > update(const common::datatypes::DetectionResults& detections);
        const std::vector<Track> &getTracks() const;
    private:
        TrackerImpl *impl_;
    };
}
