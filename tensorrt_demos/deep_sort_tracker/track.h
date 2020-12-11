#pragma once

#include "decl_spec.h"
#include "deep_sort_types.h"
#include <deque>

//forward declarations
class KalmanFilter;

namespace deep_sort {

#pragma warning(push)
#pragma warning(disable: 4251)

    class DEEP_SORT_TRACKER Track
    {
    public:

        static constexpr const int QUEUE_SIZE = 30;

        struct State {
            common::datatypes::DetectionBox position;
            common::datatypes::DetectionBox velocity;
        };

        enum TrackState {
            Tentative = 1,
            Confirmed,
            Deleted
        };

        //Base pattern to determine type of object
        enum TrackType {
            DEFAULT,
            TIME_TRACKER
        };

        Track(const common::datatypes::KalmanMeanMatType &mean, const common::datatypes::KalmanCovAMatType &covariance,
            int track_id, int n_init, int max_age, const common::datatypes::Feature &feature,
            const int class_id,
            const int max_time_since_update,
            const int max_artificial_updates
        );
        virtual ~Track() = default;

        void predit(KalmanFilter &kf);
        void update(KalmanFilter &kf, const common::datatypes::Detection &detection);
        virtual TrackType getType() const;

        void artificial_update(KalmanFilter &kf);

        void mark_missed();
        bool is_confirmed() const;
        bool is_deleted() const;
        bool is_tentative() const;

        void set_deleted();


        //TODO: refactor do not use python style in C++
        State to_tlwh() const;
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

        int artificial_updates_;
        std::deque<State> last_states_;

        int max_time_since_update_;
        int max_artificial_updates_;

    private:
        void featuresAppendOne(const common::datatypes::Feature& feature);
    };
#pragma warning(pop)
}
