#pragma once

#include "decl_spec.h"
#include "common/datatypes.h"
#include "kalman_filter.h"

namespace deep_sort {

#pragma warning(push)
#pragma warning(disable: 4251)

    class DEEP_SORT_TRACKER Track
    {
    public:
        enum TrackState {
            Tentative = 1,
            Confirmed,
            Deleted
        };

        //Base pattern to determine type of object
        enum TrackType {
            DEFAULT,
            IN_AREA_TRACKER
        };

        Track(const common::datatypes::KalmanMeanMatType &mean, const common::datatypes::KalmanCovAMatType &covariance,
            int track_id, int n_init, int max_age, const common::datatypes::Feature &feature,
            const int class_id);
        virtual ~Track() = default;

        void predit(KalmanFilter &kf);
        void update(KalmanFilter &kf, const common::datatypes::Detection &detection);
        virtual TrackType getType() const;

        void mark_missed();
        bool is_confirmed() const;
        bool is_deleted() const;
        bool is_tentative() const;


        //TODO: refactor do not use python style in C++
        common::datatypes::DetectionBox to_tlwh() const;
        int time_since_update;
        int track_id;
        int class_id;
        common::datatypes::Features features;
        common::datatypes::KalmanMeanMatType mean;
        common::datatypes::KalmanCovAMatType covariance;

        int hits;
        int age;
        int _n_init;
        int _max_age;
        TrackState state;
    private:
        void featuresAppendOne(const common::datatypes::Feature& feature);
    };
#pragma warning(pop)
}
