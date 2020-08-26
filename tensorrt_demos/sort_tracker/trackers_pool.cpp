#include "trackers_pool.h"
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

    TrackersPool::TrackersPool(int max_age, int min_hits)
        :initialized_(false)
        , max_age_(max_age)
        , min_hits_(min_hits)
    {
    }

    TrackerMatch TrackersPool::process_match(const std::vector<DetectionBox> &predicted, const DetectionResults &detections) {

        int rows = static_cast<int>(predicted.size());
        int cols = static_cast<int>(detections.size());

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
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                matrix(i, j) = 1.0 - helpers::iou(predicted[i], detections[j].bbox);
            }
        }
        Matrix<double> orig = matrix;

        //auto indices = helpers::solve(matrix);
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

    std::vector< TrackResult > TrackersPool::update(const DetectionResults& detections) {
        ++frame_counter_;

        if (!initialized_) {
            std::vector<TrackResult> out;
            for (const auto &detection : detections) {
                trackers_.push_back(SortTracker(detection.bbox, detection.class_id));
                cv::Rect cv_rect(detection.bbox(0), detection.bbox(1), detection.bbox(2), detection.bbox(3));
                out.push_back({ cv_rect, trackers_.back().getID(), detection.class_id });
            }
            initialized_ = true;
            return out;
        }

        std::vector<DetectionBox> predicted_boxes;
        for (auto it = trackers_.begin(); it != trackers_.end();)
        {
            DetectionBox pBox = (*it).predict();
            if (pBox(0) >= 0.0f && pBox(1) >= 0.0f) {
                predicted_boxes.push_back(pBox);
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

        //std::vector<int>& unmatched_tracks = res.unmatched_tracks;
        //for (int& track_idx : unmatched_tracks) {
        //    trackers_[track_idx].markMissed(max_age_);
        //}


        std::vector<int>& unmatched_detections = res.unmatched_detections;
        for (int& detection_idx : unmatched_detections) {
            trackers_.push_back(SortTracker(detections[detection_idx].bbox, detections[detection_idx].class_id));
        }

        std::vector< TrackResult > results;
        for (auto it = trackers_.begin(); it != trackers_.end();) {
            if (((*it).getTimeSinceUpdate() < 1) && (it->getHitSteak() >= min_hits_ || frame_counter_ <= min_hits_)) {
                auto rect = it->getState();
                cv::Rect cv_rect(rect(0), rect(1), rect(2), rect(3));
                results.push_back({ cv_rect , it->getID(), it->getClassID() });
                ++it;
            } else {
                ++it;
            }
            // remove dead tracklet
            if (it != trackers_.end() && it->getTimeSinceUpdate() > max_age_) {
                it = trackers_.erase(it);
            }
        }
        return results;
    }

    const std::vector<SortTracker> &TrackersPool::getTracks() const {
        return trackers_;
    }
}