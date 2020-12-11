#include "track.h"

#include <array>
#include <iostream>

namespace sort_tracker {

    using namespace common::datatypes;

    //INFO: general equation: x(t) = x0 + (xi- xi-1)dt + (xi- xi-1)dt^2/ 2
    //      but let's assume that time is constant between frames
    inline std::array<float, Track::MEASUREMENT_SIZE> convert_bound_box_to_z(const DetectionBox &bbox, const DetectionBox &prev) {
        auto x = bbox(0) + bbox(2) / 2;
        auto y = bbox(1) + bbox(3) / 2;
        auto s = bbox(2)*bbox(3);
        auto r = bbox(2) / bbox(3);
        auto prev_x = prev(0) + prev(2) / 2;
        auto prev_y = prev(1) + prev(3) / 2;
        auto vx = (x - prev_x);
        auto vy = (y - prev_y);
        return { x,y,s,r, vx, vy };
    }

    inline std::array<float, Track::MEASUREMENT_SIZE> init_bound_box_to_z(const DetectionBox &bbox) {
        auto x = bbox(0) + bbox(2) / 2;
        auto y = bbox(1) + bbox(3) / 2;
        auto s = bbox(2)*bbox(3);
        auto r = bbox(2) / bbox(3);
        return { x,y,s,r, 0.0f, 0.0f};
    }


    inline Track::State convert_xysr_to_bbox(float cx, float cy, float s, float r, float vx, float vy) {
        float w = sqrt(s * r);
        float h = s / w;
        float x = (cx - w / 2);
        float y = (cy - h / 2);

        if (x < 0 && cx > 0)
            x = 0;
        if (y < 0 && cy > 0)
            y = 0;
        return { DetectionBox(x, y, w, h), vx, vy };
    }

    int Track::ID_COUNTER = 0;

    Track::Track(const DetectionBox &bbox, const int class_id, const int time_since_update_threshold, const int first_occurance_threshold)
        :filter_(STATE_SIZE, MEASUREMENT_SIZE)
        , class_id_(class_id)
        , id_(ID_COUNTER++)
        , time_since_update_(0)
        , hits_(0)
        , hit_streak_(0)
        , age_(0)
        , max_time_since_update_threshold_(time_since_update_threshold)
        , update_counter_(0)
        , needsToDelete_(false)
        , first_occurance_threshold_(first_occurance_threshold)
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

        auto bbox_z = init_bound_box_to_z(bbox);
        for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
            filter_.statePost.at<float>(i) = bbox_z[i];
        }
        mesurements_ = cv::Mat::zeros(MEASUREMENT_SIZE, 1, CV_32FC1);

    }

    void Track::update(const common::datatypes::DetectionBox &bbox) {
        
        time_since_update_ = 0;
        ++hits_;
        ++hit_streak_;
        ++update_counter_;

        // measurement
        auto prev = history_.back();
        auto conv_z = convert_bound_box_to_z(bbox, prev);
        history_.clear();
        for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
            mesurements_.at<float>(i) = conv_z[i];
        }
        // update
        filter_.correct(mesurements_);

        auto s = filter_.statePost;
        last_states_.push_back(convert_xysr_to_bbox(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0), s.at<float>(4, 0), s.at<float>(5, 0)));
        if (last_states_.size() >= QUEUE_SIZE) {
            last_states_.pop_front();
        }
    }

    //INFO: just update the possition by the velocity vector
    void Track::aritificial_update() {
        cv::Mat p= filter_.statePost;

        if (last_states_.size() > 0) {
            auto first_state = last_states_.front();
            float first_len = std::sqrt(first_state.vx * first_state.vx + first_state.vy * first_state.vy);

            float res_vx = 0.0f;
            float res_vy = 0.0f;
            for (const auto &it : last_states_) {
                res_vx += it.vx;
                res_vy += it.vy;
            }
            float res_len = std::sqrt(res_vx*res_vx + res_vy * res_vy);
            res_vx = (res_vx / res_len) * first_len;
            res_vy = (res_vy / res_len) * first_len;

            p.at<float>(0, 0) += res_vx;
            p.at<float>(1, 0) += res_vy;
        } else {
            p.at<float>(0, 0) += p.at<float>(4, 0);
            p.at<float>(1, 0) += p.at<float>(5, 0);
        }

        for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
            mesurements_.at<float>(i) = p.at<float>(i, 0);
        }

        filter_.correct(mesurements_);

        //INFO: force to remove false tracks without detector confirmatin at first N frames state
        if (update_counter_ < first_occurance_threshold_) {
            needsToDelete_ = true;
        }
    }


    Track::State Track::predict() {
        cv::Mat p = filter_.predict();
        ++age_;

        if (time_since_update_ >= max_time_since_update_threshold_) {
            hit_streak_ = 0;
        }
        time_since_update_ += 1;

        auto rect = convert_xysr_to_bbox(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0), p.at<float>(4,0), p.at<float>(5,0));
        history_.push_back(rect.bbox);
        return rect;
    }

    Track::State Track::getState() const {
        cv::Mat s = filter_.statePost;
        auto rect = convert_xysr_to_bbox(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0), s.at<float>(4,0), s.at<float>(5,0));
        return rect;
    }

    int Track::getTimeSinceUpdate() const {
        return time_since_update_;
    }

    int Track::getID() const {
        return id_;
    }

    int Track::getHitSteak() const {
        return hit_streak_;
    }

    int Track::getClassID() const {
        return class_id_;
    }

    bool Track::needsDelete() const {
        return needsToDelete_;
    }

}