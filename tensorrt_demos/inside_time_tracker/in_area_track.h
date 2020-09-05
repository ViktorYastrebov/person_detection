#pragma once

#include "decl_spec.h"
#include "deep_sort_tracker/track.h"
#include <chrono>

namespace inside_area_tracker {
    class INSIDE_TIME_EXPORT InAreaTimeTrack : public deep_sort::Track {
    public:
        InAreaTimeTrack(const common::datatypes::KalmanMeanMatType &mean, const common::datatypes::KalmanCovAMatType &covariance,
            int track_id, int n_init, int max_age, const common::datatypes::Feature &feature,
            const int class_id);
        virtual ~InAreaTimeTrack() = default;
        virtual TrackType getType() const override;


        InAreaTimeTrack & operator +=(const std::chrono::milliseconds &t);
        std::chrono::milliseconds duration() const;
        bool isInside() const;
        void setInside(bool v);
    private:
        std::chrono::milliseconds duration_;
        bool is_inside_;
    };
}