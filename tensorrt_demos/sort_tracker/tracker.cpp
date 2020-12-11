#include "tracker_impl.h"
#include "tracker.h"

namespace sort_tracker {
    Tracker::Tracker(int max_age, int min_hits, int max_time_since_update, int false_first_occurs_limit)
        : impl_(new TrackerImpl(max_age, min_hits, max_time_since_update, false_first_occurs_limit))
    {}

    Tracker::~Tracker() {
        delete impl_;
    }

    std::vector< TrackResult > Tracker::update(const common::datatypes::DetectionResults& detections) {
        return impl_->update(detections);
    }

    const std::vector<Track> & Tracker::getTracks() const {
        return impl_->getTracks();
    }
}