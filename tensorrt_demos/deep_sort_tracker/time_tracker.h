#pragma once

#include <opencv2/opencv.hpp>
#include "tracker.h"

namespace deep_sort {

    class TimeTrackerImpl;

    class DEEP_SORT_TRACKER TimeTracker : public AbstractTracker {
    public:
        TimeTracker(const std::vector<cv::Point> &contour, float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3,
            const int max_time_since_update = 3, const int max_artificial_updates = 30);
        virtual ~TimeTracker();
        void predict() override;
        //INFO: separated function
        void update(const common::datatypes::Detections& detections, const std::chrono::milliseconds &time);
        virtual void update(const common::datatypes::Detections& detections) override;
        const std::vector<TrackPtr> &getTracks() const override;
    protected:
        TimeTrackerImpl *impl_;
    };
}