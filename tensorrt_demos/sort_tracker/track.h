#pragma once

#include "decl_spec.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/core.hpp>
#include "common/datatypes.h"
#include <deque>

namespace sort_tracker {

#pragma warning(push)
#pragma warning(disable: 4251)

    class TRACKER_ENGINE Track {
    public:

        static constexpr const int QUEUE_SIZE = 30;

        struct State {
            common::datatypes::DetectionBox bbox;
            float vx;
            float vy;
        };

        static const int STATE_SIZE = 7;
        static const int MEASUREMENT_SIZE = 6;

        Track(const common::datatypes::DetectionBox &bbox, const int class_id, const int time_since_update_threshold, const int first_occurance_threshold);
        ~Track() = default;

        void update(const common::datatypes::DetectionBox &bbox);
        void aritificial_update();
        State predict();
        State getState() const;
        int getTimeSinceUpdate() const;

        int getID() const;
        int getHitSteak() const;

        int getClassID() const;

        bool needsDelete() const;


    private:
        static int ID_COUNTER;
    private:
        cv::KalmanFilter filter_;
        cv::Mat mesurements_;
        std::vector<common::datatypes::DetectionBox> history_;

        int class_id_;

        int id_;
        int time_since_update_;

        int hits_;
        int hit_streak_;
        int age_;
        //INFO: keep last states, used for artificial_update()
        //std::array<State, 4> last_states_;
        std::deque<State> last_states_;
        int max_time_since_update_threshold_;
        int update_counter_;
        bool needsToDelete_;
        int first_occurance_threshold_;
    };
#pragma warning(pop)
}
