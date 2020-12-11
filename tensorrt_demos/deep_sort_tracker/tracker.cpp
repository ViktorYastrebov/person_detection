#include "tracker_impl.h"
#include "tracker.h"


namespace deep_sort {
    Tracker::Tracker(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init,
        const int max_time_since_update, const int max_artificial_updates)
        :impl_( new TrackerImpl(max_cosine_distance, nn_budget, max_iou_distance, max_age, n_init, max_time_since_update, max_artificial_updates))
    {}

    Tracker::~Tracker() {
        delete impl_;
    }

    void Tracker::predict() {
        impl_->predict();
    }

    void Tracker::update(const common::datatypes::Detections& detections) {
        impl_->update(detections);
    }

    const std::vector<AbstractTracker::TrackPtr> &Tracker::getTracks() const {
        return impl_->getTracks();
    }
}

