#pragma once

#include "decl_spec.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/core.hpp>
#include "datatypes.h"

namespace sort_tracker {

#pragma warning(push)
#pragma warning(disable: 4251)

    class TRACKER_ENGINE SortTracker {
    public:

        struct State {
            common::datatypes::DetectionBox bbox;
            float vx;
            float vy;
        };

        static const int STATE_SIZE = 7;
        static const int MEASUREMENT_SIZE = 6;

        SortTracker(const common::datatypes::DetectionBox &bbox, const int class_id);
        ~SortTracker() = default;

        void update(const common::datatypes::DetectionBox &bbox);
        //common::datatypes::DetectionBox predict();
        State predict();
        State getState() const;
        int getTimeSinceUpdate() const;

        int getID() const;
        int getHitSteak() const;

        int getClassID() const;

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
    };
#pragma warning(pop)
}