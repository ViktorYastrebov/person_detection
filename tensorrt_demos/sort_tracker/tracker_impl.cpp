#include "tracker_impl.h"
#include "common/hungarian_eigen/hungarian.h"
#include <numeric>

namespace sort_tracker {

    using namespace common::datatypes;
    namespace helpers {
        double iou(const DetectionBox &lbox, const DetectionBox &rbox) {
            float interBox[] = {
                std::max(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
                std::min(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
                std::max(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
                std::min(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
            };

            if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
                return 0.0;

            float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
            return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
        }
    }

    TrackerImpl::TrackerImpl(int max_age, int min_hits, int max_time_since_update, int false_first_occurs_limit)
        :initialized_(false)
        , max_age_(max_age)
        , min_hits_(min_hits)
        , max_time_since_update_(max_time_since_update)
        , first_false_occurance_limit_(false_first_occurs_limit)
    {
    }

    TrackerMatch TrackerImpl::process_match(const std::vector<DetectionBox> &predicted, const DetectionResults &detections) {

        auto rows = static_cast<int>(predicted.size());
        auto cols = static_cast<int>(detections.size());

        std::vector<int> detection_idxs(detections.size());
        std::iota(detection_idxs.begin(), detection_idxs.end(), 0);
        std::vector<int> track_idxs(predicted.size());
        std::iota(track_idxs.begin(), track_idxs.end(), 0);

        //INFO: return unmatched all for empty values
        if ((detection_idxs.size() == 0) || (track_idxs.size() == 0)) {
            TrackerMatch res;
            res.unmatched_tracks.assign(track_idxs.begin(), track_idxs.end());
            res.unmatched_detections.assign(detection_idxs.begin(), detection_idxs.end());
            return res;
        }

        TrackerMatch res;
        Matrix<double> matrix(rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                matrix(i, j) = 1.0 - helpers::iou(predicted[i], detections[j].bbox);
            }
        }
        Matrix<double> orig = matrix;

        auto indices = hungarian::solve(matrix);

        for (size_t col = 0; col < detection_idxs.size(); col++) {
            bool flag = false;
            for (int i = 0; i < indices.rows(); i++)
                if (indices(i, 1) == col) {
                    flag = true;
                    break;
                }
            if (!flag) {
                res.unmatched_detections.push_back(detection_idxs[col]);
            }
        }

        for (size_t row = 0; row < track_idxs.size(); row++) {
            bool flag = false;
            for (int i = 0; i < indices.rows(); i++)
                if (indices(i, 0) == row) {
                    flag = true;
                    break;
                }
            if (!flag) {
                res.unmatched_tracks.push_back(track_idxs[row]);
            }
        }

        for (int i = 0; i < indices.rows(); i++) {
            int row = indices(i, 0);
            int col = indices(i, 1);

            int track_idx = track_idxs[row];
            int detection_idx = detection_idxs[col];
            double process = 1.0 - orig(row, col);
            if(process < iou_threshold_)
            {
                res.unmatched_tracks.push_back(track_idx);
                res.unmatched_detections.push_back(detection_idx);
            } else {
                res.matches.push_back(std::make_pair(track_idx, detection_idx));
            }
        }
        return res;
    }

    std::vector< TrackResult > TrackerImpl::update(const DetectionResults& detections) {
        ++frame_counter_;

        if (!initialized_) {
            std::vector<TrackResult> out;
            for (const auto &detection : detections) {
                trackers_.push_back(Track(detection.bbox, detection.class_id, 2* max_time_since_update_, first_false_occurance_limit_));
                cv::Rect cv_rect(detection.bbox(0), detection.bbox(1), detection.bbox(2), detection.bbox(3));
                out.push_back({ cv_rect, trackers_.back().getID(), detection.class_id, 0.0f, 0.0f });
            }
            initialized_ = true;
            return out;
        }

        std::vector<DetectionBox> predicted_boxes;
        for (auto it = trackers_.begin(); it != trackers_.end();)
        {
            //INFO: need to add filter that out of the bounds
            auto state = (*it).predict();
            if (state.bbox(0) >= 0.0f && state.bbox(1) >= 0.0f) {
                predicted_boxes.push_back(state.bbox);
                ++it;
            } else {
                it = trackers_.erase(it);
            }
        }

        TrackerMatch res = process_match(predicted_boxes, detections);
        std::vector<MatchData>& matches = res.matches;

        for (MatchData& data : matches) {
            int track_idx = data.first;
            int detection_idx = data.second;
            trackers_[track_idx].update(detections[detection_idx].bbox);
        }

        //INFO: here is the case:
        //      1. Fails of detector produce the new Track.
        //        and artificial_updating keeps it but no real detection has happened
        //      2. Track is real
        //         at some point detector fails and need to keep the track some time.
        std::vector<int>& unmatched_tracks = res.unmatched_tracks;
        for (int& track_idx : unmatched_tracks) {
            //INFO: can add artificial updating
            trackers_[track_idx].aritificial_update();
        }


        std::vector<int>& unmatched_detections = res.unmatched_detections;
        for (int& detection_idx : unmatched_detections) {
            trackers_.push_back(Track(detections[detection_idx].bbox, detections[detection_idx].class_id, 2* max_time_since_update_, first_false_occurance_limit_));
        }

        std::vector< TrackResult > results;
        for (auto it = trackers_.begin(); it != trackers_.end();) {
            if (((*it).getTimeSinceUpdate() < max_time_since_update_) && (it->getHitSteak() >= min_hits_ || frame_counter_ <= min_hits_)) {
                auto state = it->getState();
                cv::Rect cv_rect(state.bbox(0), state.bbox(1), state.bbox(2), state.bbox(3));
                results.push_back({ cv_rect , it->getID(), it->getClassID(), state.vx, state.vy});
                ++it;
            } else {
                ++it;
            }
            // remove dead tracklet
            if (it != trackers_.end() && (it->getTimeSinceUpdate() > max_age_ || it->needsDelete())) {
                it = trackers_.erase(it);
            }
        }
        return results;
    }

    const std::vector<Track> &TrackerImpl::getTracks() const {
        return trackers_;
    }
}