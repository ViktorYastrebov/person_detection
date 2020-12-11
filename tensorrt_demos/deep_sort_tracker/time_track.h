#pragma once

#include "decl_spec.h"
#include "track.h"
#include <chrono>
#include "track.h"

namespace deep_sort {
#pragma warning(push)
#pragma warning(disable: 4251)

    class DEEP_SORT_TRACKER TimeTrack : public Track {
    public:
        TimeTrack(const common::datatypes::KalmanMeanMatType &mean, const common::datatypes::KalmanCovAMatType &covariance,
            int track_id, int n_init, int max_age, const common::datatypes::Feature &feature,
            const int class_id,
            const int max_time_since_update,
            const int max_artificial_updates
        );
        virtual ~TimeTrack() = default;
        virtual TrackType getType() const override;


        TimeTrack & operator +=(const std::chrono::milliseconds &t);
        std::chrono::milliseconds duration() const;
        bool isInside() const;
        void setInside(bool v);
    private:
        std::chrono::milliseconds duration_;
        bool is_inside_;
    };
#pragma warning(pop)
}