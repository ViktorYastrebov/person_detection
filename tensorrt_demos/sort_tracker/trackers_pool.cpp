#include "trackers_pool.h"

#include "common/hungarian_eigen/munkres/munkres.h"
#include "Hungarian.h"
#include <set>
#include <numeric>

namespace sort_tracker {

    using namespace common::datatypes;
#if 0
    using DetectionBoxes = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;
    namespace helper {

        Eigen::VectorXf iou(DetectionBox& bbox, DetectionBoxes &candidates) {
            float bbox_tl_1 = bbox[0];
            float bbox_tl_2 = bbox[1];
            float bbox_br_1 = bbox[0] + bbox[2];
            float bbox_br_2 = bbox[1] + bbox[3];
            float area_bbox = bbox[2] * bbox[3];

            Eigen::Matrix<float, -1, 2> candidates_tl;
            Eigen::Matrix<float, -1, 2> candidates_br;
            candidates_tl = candidates.leftCols(2);
            candidates_br = candidates.rightCols(2) + candidates_tl;

            int size = int(candidates.rows());
            Eigen::VectorXf res(size);
            for (int i = 0; i < size; i++) {
                float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
                float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
                float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
                float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

                float w = br_1 - tl_1; w = (w < 0 ? 0 : w);
                float h = br_2 - tl_2; h = (h < 0 ? 0 : h);
                float area_intersection = w * h;
                float area_candidates = candidates(i, 2) * candidates(i, 3);
                res[i] = area_intersection / (area_bbox + area_candidates - area_intersection);
            }
            return res;
        }

        CostMatrixType iou_cost(
            std::vector<SortTracker>& tracks,
            const DetectionResults& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices)
        {

            //!!!python diff: track_indices && detection_indices will never be None.
        //    if(track_indices.empty() == true) {
        //        for(size_t i = 0; i < tracks.size(); i++) {
        //            track_indices.push_back(i);
        //        }
        //    }
        //    if(detection_indices.empty() == true) {
        //        for(size_t i = 0; i < dets.size(); i++) {
        //            detection_indices.push_back(i);
        //        }
        //    }
            int rows = track_indices.size();
            int cols = detection_indices.size();
            CostMatrixType cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
            for (int i = 0; i < rows; i++) {
                int track_idx = track_indices[i];
                if (tracks[track_idx].getTimeSinceUpdate() > 1) {
                    cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
                    continue;
                }
                //auto bbox = tracks[track_idx].to_tlwh();
                auto bbox = tracks[track_idx].getState();
                int csize = detection_indices.size();
                DetectionBoxes candidates(csize, 4);
                for (int k = 0; k < csize; k++) {
                    //candidates.row(k) = dets[detection_indices[k]].tlwh;
                    candidates.row(k) = dets[detection_indices[k]].bbox;
                }
                Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
                cost_matrix.row(i) = rowV;
            }
            return cost_matrix;
        }

    }
#endif

    namespace helpers {
        double GetIOU(const cv::Rect &bb_test, const cv::Rect &bb_gt) {
            float in = static_cast<float>((bb_test & bb_gt).area());
            float un = bb_test.area() + bb_gt.area() - in;
            //TODO: check for close function
            if (un < DBL_EPSILON)
                return 0.0;
            return (double)(in / un);
        }

        double iou(const DetectionBox &lbox, const DetectionBox &rbox) {
            float interBox[] = {
                std::max(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
                std::min(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
                std::max(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
                std::min(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
            };

            if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
                return 0.0f;

            float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
            return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
        }

        Eigen::Matrix<float, -1, 2, Eigen::RowMajor> solve(Matrix<double> &matrix, const int rows, const int cols) {
            Munkres<double> m;
            m.solve(matrix);

            std::vector<std::pair<int, int>> pairs;
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    int tmp = (int)matrix(row, col);
                    if (tmp == 0) pairs.push_back(std::make_pair(row, col));
                }
            }
            //
            int count = pairs.size();
            Eigen::Matrix<float, -1, 2, Eigen::RowMajor> re(count, 2);
            for (int i = 0; i < count; i++) {
                re(i, 0) = pairs[i].first;
                re(i, 1) = pairs[i].second;
            }
            return re;
        }

    }

    TrackersPool::TrackersPool(int max_age, int min_hits)
        :initialized_(false)
        , max_age_(max_age)
        , min_hits_(min_hits)
    {}

    TrackerMatch TrackersPool::process_match(const std::vector<DetectionBox> &predicted, const DetectionResults &detections) {

        int n_predicted = static_cast<int>(predicted.size());
        int n_detected = static_cast<int>(detections.size());

        std::vector<int> track_idxs(n_predicted);
        std::vector<int> detection_idxs(n_detected);

        std::iota(track_idxs.begin(), track_idxs.end(), 0);
        std::iota(detection_idxs.begin(), detection_idxs.end(), 0);

        TrackerMatch res;
        Matrix<double> matrix(n_predicted, n_detected);
        for (unsigned int i = 0; i < n_predicted; ++i) {
            for (unsigned int j = 0; j < n_detected; ++j) {
                matrix(i, j) = 1.0 - helpers::iou(predicted[i], detections[j].bbox);
            }
        }
        Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = helpers::solve(matrix, n_predicted, n_detected);

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
                //unmatched_tracks.push_back(row);
                res.unmatched_tracks.push_back(track_idxs[row]);
            }
        }

        for (int i = 0; i < indices.rows(); i++) {
            int row = indices(i, 0);
            int col = indices(i, 1);

            int track_idx = track_idxs[row];
            int detection_idx = detection_idxs[col];
            //if (1 - iouMatrix[i][assignment[i]] < iou_threshold_) {
            double process = 1.0 - matrix(row, col);
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


    void TrackersPool::update(const DetectionResults& detections) {
        ++frame_counter_;

        if (!initialized_) {
            //std::vector<TrackResult> out;
            for (const auto &detection : detections) {
                trackers_.push_back(SortTracker(detection.bbox, detection.class_id));
                //out.push_back({ detection.bbox, trackers_.back().getID(), detection.class_id });
            }
            initialized_ = true;
            //return out;
            return;
        }

        //INFO: here it does not take into account any hyper params at all ?????
        std::vector<DetectionBox> predicted_boxes;
        for (auto it = trackers_.begin(); it != trackers_.end();)
        {
            //MIGHT NEED TO MAKE DETECTION RESULT
            DetectionBox pBox = (*it).predict();
            if (pBox(0) >= 0.0f && pBox(1) >= 0.0f) {
                predicted_boxes.push_back(pBox);
                //predicted_idxs.push_back(track_idx);
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
            //tracks[track_idx].update(*kalman_filter, detections[detection_idx]);
        }

        //REMOVE
        std::vector<int>& unmatched_tracks = res.unmatched_tracks;
        for (int& track_idx : unmatched_tracks) {
            //if (it != trackers_.end() && it->getTimeSinceUpdate() > max_age_) {
            if (trackers_[track_idx].getTimeSinceUpdate() > max_age_) {
                trackers_[track_idx].setDeleted();
            }
            //this->tracks[track_idx].mark_missed();
        }
        std::vector<int>& unmatched_detections = res.unmatched_detections;
        for (int& detection_idx : unmatched_detections) {
            trackers_.push_back(SortTracker(detections[detection_idx].bbox, detections[detection_idx].class_id));
        }

        std::vector<SortTracker>::iterator it = trackers_.begin();
        for (it; it != trackers_.end();) {
            if ((*it).isDeleted()) {
                it = trackers_.erase(it);
            } else {
                ++it;
            }
        }
    }

    const std::vector<SortTracker> &TrackersPool::getTracks() const {
        return trackers_;
    }

#if 0
    std::vector<TrackResult> TrackersPool::update(const DetectionResults &detections) {
        ++frame_counter_;

        if (!initialized_) {
            std::vector<TrackResult> out;
            for (const auto &detection : detections) {
                trackers_.push_back(SortTracker(detection.bbox, detection.class_id));
                //INFO: test
                cv::Rect cv_rect(detection.bbox(0), detection.bbox(1), detection.bbox(2), detection.bbox(3));
                out.push_back({ cv_rect , trackers_.back().getID(), detection.class_id });
            }
            initialized_ = true;
            return out;
        }

        //std::vector<cv::Rect> predicted_boxes;
        std::vector<common::datatypes::DetectionBox> predicted_boxes;
        for (auto it = trackers_.begin(); it != trackers_.end();)
        {
            //cv::Rect pBox = (*it).predict();
            common::datatypes::DetectionBox pBox = (*it).predict();
            if (pBox(0) >= 0 && pBox(1) >= 0) {
                predicted_boxes.push_back(pBox);
                ++it;
            } else {
                it = trackers_.erase(it);
            }
        }
        std::size_t n_predicted = predicted_boxes.size();
        std::size_t n_detected = detections.size();

        std::vector<std::vector<double>> iouMatrix(n_predicted, std::vector<double>(n_detected, 0));

        for (unsigned int i = 0; i < n_predicted; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < n_detected; j++)
            {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1.0 - helpers:(predicted_boxes[i], detections[j].bbox);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        std::vector<int> assignment;
        HungAlgo.Solve(iouMatrix, assignment);

        std::set<int> allItems;
        std::set<int> matchedItems;
        std::set<int> unmatchedDetections;
        std::set<int> unmatchedTrajectories;
        std::vector<cv::Point> matchedPairs;

        // INFO: there are unmatched detections
        if (n_detected > n_predicted) {
            for (unsigned int n = 0; n < n_detected; ++n) {
                allItems.insert(n);
            }

            for (unsigned int i = 0; i < n_predicted; ++i) {
                matchedItems.insert(assignment[i]);
            }

            set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(),
                std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        } else {
            // there are unmatched trajectory/predictions
            if (n_detected < n_predicted) {
                for (unsigned int i = 0; i < n_predicted; ++i) {
                    // unassigned label will be set as -1 in the assignment algorithm
                    if (assignment[i] == -1) {
                        unmatchedTrajectories.insert(i);
                    }
                }
            }
        }

        for (unsigned int i = 0; i < n_predicted; ++i) {
            // pass over invalid values
            if (assignment[i] == -1) {
                continue;
            }
            if (1 - iouMatrix[i][assignment[i]] < iou_threshold_) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            } else {
                matchedPairs.push_back(cv::Point(i, assignment[i]));
            }
        }

        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); ++i) {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers_[trkIdx].update(detections[detIdx].bbox);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections) {
            //KalmanTracker tracker = KalmanTracker(detections[umd]);
            //trackers.push_back(tracker);
            trackers_.push_back(SortTracker(detections[umd].bbox, detections[umd].class_id));
        }

        std::vector< TrackResult > results;
        for (auto it = trackers_.begin(); it != trackers_.end();) {
            if (((*it).getTimeSinceUpdate() < 1) && (it->getHitSteak() >= min_hits_ || frame_counter_ <= min_hits_)) {
                //INFO: test
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
#endif

#if 0
    void TrackersPool::update(const std::vector<DetectionResult> &detections) {
        ++frame_counter_;

        if (!initialized_) {
            std::vector<TrackResult> out;
            for (const auto &detection : detections) {
                trackers_.push_back(KalmanTracker(detection.bbox, detection.class_id));
                out.push_back({ detection.bbox, trackers_.back().getID(), detection.class_id });
            }
            initialized_ = true;
            return out;
        }

        std::vector<cv::Rect> predicted_boxes;
        for (auto it = trackers_.begin(); it != trackers_.end();)
        {
            cv::Rect pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0) {
                predicted_boxes.push_back(pBox);
                ++it;
            } else {
                it = trackers_.erase(it);
            }
        }
        std::size_t n_predicted = predicted_boxes.size();
        std::size_t n_detected = detections.size();

        std::vector<std::vector<double>> iouMatrix(n_predicted, std::vector<double>(n_detected, 0));


        for (unsigned int i = 0; i < n_predicted; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < n_detected; j++)
            {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1.0 - helpers::GetIOU(predicted_boxes[i], detections[j].bbox);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        std::vector<int> assignment;
        HungAlgo.Solve(iouMatrix, assignment);

        std::set<int> allItems;
        std::set<int> matchedItems;
        std::set<int> unmatchedDetections;
        std::set<int> unmatchedTrajectories;
        std::vector<cv::Point> matchedPairs;

        // INFO: there are unmatched detections
        if (n_detected > n_predicted) {
            for (unsigned int n = 0; n < n_detected; ++n) {
                allItems.insert(n);
            }

            for (unsigned int i = 0; i < n_predicted; ++i) {
                matchedItems.insert(assignment[i]);
            }

            set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(),
                            std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        } else {
            // there are unmatched trajectory/predictions
            if (n_detected < n_predicted) {
                for (unsigned int i = 0; i < n_predicted; ++i) {
                    // unassigned label will be set as -1 in the assignment algorithm
                    if (assignment[i] == -1) {
                        unmatchedTrajectories.insert(i);
                    }
                }
            }
        }

        for (unsigned int i = 0; i < n_predicted; ++i) {
            // pass over invalid values
            if (assignment[i] == -1) {
                continue;
            }
            if (1 - iouMatrix[i][assignment[i]] < iou_threshold_) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            } else {
                matchedPairs.push_back(cv::Point(i, assignment[i]));
            }
        }

        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); ++i) {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers_[trkIdx].update(detections[detIdx].bbox);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections) {
            //KalmanTracker tracker = KalmanTracker(detections[umd]);
            //trackers.push_back(tracker);
            trackers_.push_back(KalmanTracker(detections[umd].bbox, detections[umd].class_id));
        }

        std::vector< TrackResult > results;
        for (auto it = trackers_.begin(); it != trackers_.end();) {
            if (((*it).getTimeSinceUpdate() < 1) && (it->getHitSteak() >= min_hits_ || frame_counter_ <= min_hits_)) {
                results.push_back({ it->getState(), it->getID(), it->getClassID() });
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
#endif

}