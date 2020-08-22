#include "kalman_tracker.h"

#include <array>
#include <iostream>

namespace sort_tracker {

    using namespace common::datatypes;


#if 0
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

#else
    inline std::array<float, SortTracker::MEASUREMENT_SIZE> convert_bound_box_to_z(const cv::Rect &bbox) {
        auto x = static_cast<float>(bbox.x) + static_cast<float>(bbox.width) / 2;
        auto y = static_cast<float>(bbox.y) + static_cast<float>(bbox.height) / 2;
        // INFO: scale is just area
        auto s = static_cast<float>(bbox.area());
        auto r = bbox.width / static_cast<float>(bbox.height);
        return {x,y,s,r};
    }


    inline cv::Rect convert_xysr_to_bbox(float cx, float cy, float s, float r) {
        float w = sqrt(s * r);
        float h = s / w;
        float x = (cx - w / 2);
        float y = (cy - h / 2);

        if (x < 0 && cx > 0)
            x = 0;
        if (y < 0 && cy > 0)
            y = 0;
        //TODO: add round
        return cv::Rect(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h));
    }
#endif

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

        //INFO: test
        cv::Rect cv_bbox(bbox(0), bbox(1), bbox(2), bbox(3));

        auto bbox_z = convert_bound_box_to_z(cv_bbox);
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

        //INFO: test
        cv::Rect cv_bbox(bbox(0), bbox(1), bbox(2), bbox(3));

        // measurement
        auto conv_z = convert_bound_box_to_z(cv_bbox);
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
        //INFO: test
        common::datatypes::DetectionBox eigen_bbox(rect.x, rect.y, rect.width, rect.height);
        history_.push_back(eigen_bbox);
        return history_.back();
    }

    common::datatypes::DetectionBox SortTracker::getState() const {
        cv::Mat s = filter_.statePost;
        auto cv_rect = convert_xysr_to_bbox(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
        //INFO: test
        common::datatypes::DetectionBox eigen_bbox(cv_rect.x, cv_rect.y, cv_rect.width, cv_rect.height);
        return eigen_bbox;
    }

#if 0
    void SortTracker::update(const cv::Rect &bbox) {
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

    cv::Rect SortTracker::predict() {

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

    cv::Rect SortTracker::getState() const {
        cv::Mat s = filter_.statePost;
        return convert_xysr_to_bbox(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
    }
#endif

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

    void SortTracker::setDeleted() {
        is_deleted_ = true;
    }

    bool SortTracker::isDeleted() const {
        return is_deleted_;
    }
}