#pragma once

#include "decl_spec.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/core.hpp>

namespace tracker {

#pragma warning(push)
#pragma warning(disable: 4251)

    class TRACKER_ENGINE KalmanTracker {
    public:

        static const int STATE_SIZE = 7;
        static const int MEASUREMENT_SIZE = 4;

        KalmanTracker(const cv::Rect &bbox, const int class_id);
        ~KalmanTracker() = default;

        void update(const cv::Rect &bbox);
        cv::Rect predict();
        cv::Rect getState() const;
        int getTimeSinceUpdate() const;

        int getID() const;
        int getHitSteak() const;

        int getClassID() const;

    private:
        static int ID_COUNTER;
    private:
        cv::KalmanFilter filter_;
        cv::Mat mesurements_;
        std::vector<cv::Rect> history_;

        int class_id_;

        int id_;
        int time_since_update_;

        int hits_;
        int hit_streak_;
        int age_;
    };
#pragma warning(pop)
}