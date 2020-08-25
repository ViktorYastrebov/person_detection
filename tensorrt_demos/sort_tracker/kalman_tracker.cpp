#include "kalman_tracker.h"

#include <array>
#include <iostream>

namespace sort_tracker {

    using namespace common::datatypes;

    inline std::array<float, SortTracker::MEASUREMENT_SIZE> convert_bound_box_to_z(const DetectionBox &bbox) {
        auto x = bbox(0) + bbox(2) / 2;
        auto y = bbox(1) + bbox(3) / 2;
        auto s = bbox(2)*bbox(3);
        auto r = bbox(2) / bbox(3);
        return { x,y,s,r };
    }

    inline DetectionBox convert_xysr_to_bbox(float cx, float cy, float s, float r) {
        float w = sqrt(s * r);
        float h = s / w;
        float x = (cx - w / 2);
        float y = (cy - h / 2);

        if (x < 0 && cx > 0)
            x = 0;
        if (y < 0 && cy > 0)
            y = 0;
        return DetectionBox(x, y, w, h);
    }

    int SortTracker::ID_COUNTER = 0;

    SortTracker::SortTracker(const DetectionBox &bbox, const int class_id)
        :filter_(STATE_SIZE, MEASUREMENT_SIZE)
        , class_id_(class_id)
        , id_(ID_COUNTER++)
        , time_since_update_(0)
        , hits_(0)
        , hit_streak_(0)
        , age_(0)
    {
        cv::setIdentity(filter_.transitionMatrix);

        cv::setIdentity(filter_.measurementMatrix);

        cv::setIdentity(filter_.processNoiseCov, cv::Scalar::all(1));
        filter_.processNoiseCov.at<float>(4, 4) = 0.01f;
        filter_.processNoiseCov.at<float>(5, 5) = 0.01f;
        filter_.processNoiseCov.at<float>(6, 6) = 0.0001f;

        cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(1.0));
        filter_.measurementNoiseCov.at<float>(2, 2) = 10.0f;
        filter_.measurementNoiseCov.at<float>(3, 3) = 10.0f;

        cv::setIdentity(filter_.errorCovPost, cv::Scalar::all(10)); // P
        filter_.errorCovPost.at<float>(4, 4) *= 1000;
        filter_.errorCovPost.at<float>(5, 5) *= 1000;
        filter_.errorCovPost.at<float>(6, 6) *= 1000;

        auto bbox_z = convert_bound_box_to_z(bbox);
        for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
            filter_.statePost.at<float>(i) = bbox_z[i];
        }
        mesurements_ = cv::Mat::zeros(MEASUREMENT_SIZE, 1, CV_32FC1);

    }

    void SortTracker::update(const common::datatypes::DetectionBox &bbox) {
        time_since_update_ = 0;
        history_.clear();
        ++hits_;
        ++hit_streak_;

        // measurement
        auto conv_z = convert_bound_box_to_z(bbox);
        for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
            mesurements_.at<float>(i) = conv_z[i];
        }
        // update
        filter_.correct(mesurements_);
    }

    common::datatypes::DetectionBox SortTracker::predict() {
        cv::Mat p = filter_.predict();
        ++age_;

        if (time_since_update_ > 0) {
            hit_streak_ = 0;
        }
        time_since_update_ += 1;

        auto rect = convert_xysr_to_bbox(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));
        history_.push_back(rect);
        return history_.back();
    }

    common::datatypes::DetectionBox SortTracker::getState() const {
        cv::Mat s = filter_.statePost;
        auto rect = convert_xysr_to_bbox(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
        return rect;
    }

    int SortTracker::getTimeSinceUpdate() const {
        return time_since_update_;
    }

    int SortTracker::getID() const {
        return id_;
    }

    int SortTracker::getHitSteak() const {
        return hit_streak_;
    }

    int SortTracker::getClassID() const {
        return class_id_;
    }
}