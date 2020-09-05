#include "in_area_track.h"

namespace inside_area_tracker {
    using namespace deep_sort;

    InAreaTimeTrack::InAreaTimeTrack(const common::datatypes::KalmanMeanMatType &mean,
        const common::datatypes::KalmanCovAMatType &covariance,
        int track_id,
        int n_init,
        int max_age,
        const common::datatypes::Feature &feature,
        const int class_id)
        :Track(mean, covariance, track_id, n_init, max_age, feature, class_id)
        , duration_()
        , is_inside_(false)
    {}

    Track::TrackType InAreaTimeTrack::getType() const {
        return TrackType::IN_AREA_TRACKER;
    }

    InAreaTimeTrack & InAreaTimeTrack::operator +=(const std::chrono::milliseconds &t) {
        if (is_inside_) {
            duration_ += t;
        }
        return *this;
    }

    std::chrono::milliseconds InAreaTimeTrack::duration() const {
        return duration_;
    }

    bool InAreaTimeTrack::isInside() const {
        return is_inside_;
    }

    void InAreaTimeTrack::setInside(bool v) {
        is_inside_ = v;
        if (!is_inside_) {
            duration_ = std::chrono::milliseconds(0);
        }
    }

}