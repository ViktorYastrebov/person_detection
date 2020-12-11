#include "time_tracker_impl.h"
#include "time_tracker.h"


namespace deep_sort {

    TimeTracker::TimeTracker(const std::vector<cv::Point> &contour, float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init,
                             const int max_time_since_update, const int max_artificial_updates)
        : impl_(new TimeTrackerImpl(contour, max_cosine_distance, nn_budget, max_iou_distance, max_age, n_init, max_time_since_update, max_artificial_updates))
    {
    }

    TimeTracker::~TimeTracker() {
        delete impl_;
    }

    void TimeTracker::predict() {
        impl_->predict();
    }

    void TimeTracker::update(const common::datatypes::Detections& detections, const std::chrono::milliseconds &time) {
        impl_->update(detections, time);
    }

    void TimeTracker::update(const common::datatypes::Detections& detections)
    {}

    const std::vector<AbstractTracker::TrackPtr> &TimeTracker::getTracks() const {
        return impl_->getTracks();
    }
}