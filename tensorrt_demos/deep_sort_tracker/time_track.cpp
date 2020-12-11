#include "time_track.h"

namespace deep_sort {

    TimeTrack::TimeTrack(
        const common::datatypes::KalmanMeanMatType &mean,
        const common::datatypes::KalmanCovAMatType &covariance,
        int track_id,
        int n_init,
        int max_age,
        const common::datatypes::Feature &feature,
        const int class_id,
        const int max_time_since_update,
        const int max_artificial_updates
    )
        :Track(mean, covariance, track_id, n_init, max_age, feature, class_id, max_time_since_update, max_artificial_updates)
        , duration_()
        , is_inside_(false)
    {}

    Track::TrackType TimeTrack::getType() const {
        return TrackType::TIME_TRACKER;
    }

    TimeTrack & TimeTrack::operator +=(const std::chrono::milliseconds &t) {
        if (is_inside_) {
            duration_ += t;
        }
        return *this;
    }

    std::chrono::milliseconds TimeTrack::duration() const {
        return duration_;
    }

    bool TimeTrack::isInside() const {
        return is_inside_;
    }

    void TimeTrack::setInside(bool v) {
        is_inside_ = v;
        if (!is_inside_) {
            duration_ = std::chrono::milliseconds(0);
        }
    }

}