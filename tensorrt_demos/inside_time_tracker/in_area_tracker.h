#pragma once

#include <opencv2/opencv.hpp>
#include "decl_spec.h"
#include "deep_sort_tracker/tracker.h"

namespace inside_area_tracker {

    class INSIDE_TIME_EXPORT InAreaTracker : public deep_sort::Tracker {
    public:
        InAreaTracker(const std::vector<cv::Point> &contour, float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3);
        virtual ~InAreaTracker() = default;

        virtual void predict() override;
        virtual void update(const common::datatypes::Detections& detections, const std::chrono::milliseconds &time);

    private:
        TrackPtr initialize_track(
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