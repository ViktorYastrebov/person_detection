#pragma once

#include <opencv2/opencv.hpp>
#include "tracker_impl.h"

namespace deep_sort {

    class TimeTrackerImpl : public TrackerImpl {
    public:
        TimeTrackerImpl(const std::vector<cv::Point> &contour,
            float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3,
            const int max_time_since_update = 3, const int max_artificial_updates = 30);
        virtual ~TimeTrackerImpl() = default;

        virtual void predict();
        virtual void update(const common::datatypes::Detections& detections, const std::chrono::milliseconds &time);

    private:
        AbstractTracker::TrackPtr initialize_track(
            const common::datatypes::KalmanMeanMatType &mean,
            const common::datatypes::KalmanCovAMatType &covariance,
            int track_id,
            int n_init,
            int max_age,
            const common::datatypes::Feature &feature,
            const int class_id);
    private:
        std::vector<cv::Point> contour_;
    };
}