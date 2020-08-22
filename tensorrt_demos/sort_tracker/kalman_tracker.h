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

        static const int STATE_SIZE = 7;
        static const int MEASUREMENT_SIZE = 4;

        //KalmanTracker(const cv::Rect &bbox, const int class_id);
        SortTracker(const common::datatypes::DetectionBox &bbox, const int class_id);
        ~SortTracker() = default;

        void update(const common::datatypes::DetectionBox &bbox);
        common::datatypes::DetectionBox predict();
        common::datatypes::DetectionBox getState() const;
#if 0
        void update(const cv::Rect &bbox);
        cv::Rect predict();
        cv::Rect getState() const;
#endif
        int getTimeSinceUpdate() const;

        int getID() const;
        int getHitSteak() const;

        int getClassID() const;

        void setDeleted();
        bool isDeleted() const;

    private:
        static int ID_COUNTER;
    private:
        //TODO: check with Eigen implementation !!!
        //      It allows to generalize it
        cv::KalmanFilter filter_;
        cv::Mat mesurements_;
        //std::vector<cv::Rect> history_;
        std::vector<common::datatypes::DetectionBox> history_;

        int class_id_;

        int id_;
        int time_since_update_;

        int hits_;
        int hit_streak_;
        int age_;
        bool is_deleted_ = false;

    };
#pragma warning(pop)
}