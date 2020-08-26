#include "linear_assignment.h"
//#include "hungarian_eigen/hungarian.h"
#include "common/hungarian_eigen/hungarian.h"
#include <map>

namespace linear_assignment {

    using namespace common::datatypes;
    using namespace deep_sort;

    const double chi2inv95[10] = {
        0,
        3.8415,
        5.9915,
        7.8147,
        9.4877,
        11.070,
        12.592,
        14.067,
        15.507,
        16.919
    };


    TrackerMatch matching_cascade(Tracker* distance_metric,
        Tracker::MetricFunction metric_function,
        float max_distance,
        int cascade_depth,
        std::vector<Track>& tracks,
        const Detections &detections,
        std::vector<int> &track_indices,
        std::vector<int> detection_indices)
    {
        TrackerMatch res;
        //!!!python diff: track_indices will never be None.
        //    if(track_indices.empty() == true) {
        //        for(size_t i = 0; i < tracks.size(); i++) {
        //            track_indices.push_back(i);
        //        }
        //    }

        //!!!python diff: detection_indices will always be None.
        for (size_t i = 0; i < detections.size(); i++) {
            detection_indices.push_back(int(i));
        }

        std::vector<int> unmatched_detections;
        unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
        res.matches.clear();
        std::vector<int> track_indices_l;

        std::map<int, int> matches_trackid;
        for (int level = 0; level < cascade_depth; level++) {
            if (unmatched_detections.size() == 0) break; //No detections left;

            track_indices_l.clear();
            for (int k : track_indices) {
                if (tracks[k].time_since_update == 1 + level)
                    track_indices_l.push_back(k);
            }
            if (track_indices_l.size() == 0) continue; //Nothing to match at this level.


            TrackerMatch tmp = min_cost_matching(
                distance_metric,
                metric_function,
                max_distance, tracks, detections, track_indices_l,
                unmatched_detections);

            unmatched_detections.assign(tmp.unmatched_detections.begin(), tmp.unmatched_detections.end());
            for (size_t i = 0; i < tmp.matches.size(); i++) {
                MatchData pa = tmp.matches[i];
                res.matches.push_back(pa);
                matches_trackid.insert(pa);
            }
        }
        res.unmatched_detections.assign(unmatched_detections.begin(), unmatched_detections.end());
        for (size_t i = 0; i < track_indices.size(); i++) {
            int tid = track_indices[i];
            if (matches_trackid.find(tid) == matches_trackid.end())
                res.unmatched_tracks.push_back(tid);
        }
        return res;
    }

    TrackerMatch min_cost_matching(
        Tracker* distance_metric,
        Tracker::MetricFunction metric_function,
        float max_distance,
        std::vector<Track>& tracks,
        const Detections &detections,
        std::vector<int>& track_indices,
        std::vector<int>& detection_indices)
    {
        TrackerMatch res;
        //!!!python diff: track_indices && detection_indices will never be None.
        //    if(track_indices.empty() == true) {
        //        for(size_t i = 0; i < tracks.size(); i++) {
        //            track_indices.push_back(i);
        //        }
        //    }
        //    if(detection_indices.empty() == true) {
        //        for(size_t i = 0; i < detections.size(); i++) {
        //            detection_indices.push_back(int(i));
        //        }
        //    }
        if ((detection_indices.size() == 0) || (track_indices.size() == 0)) {
            res.matches.clear();
            res.unmatched_tracks.assign(track_indices.begin(), track_indices.end());
            res.unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
            return res;
        }

        constexpr const float EPSILON = 1e-5f;
        CostMatrixType cost_matrix = (distance_metric->*(metric_function)) (tracks, detections, track_indices, detection_indices);

        for (int i = 0; i < cost_matrix.rows(); i++) {
            for (int j = 0; j < cost_matrix.cols(); j++) {
                float tmp = cost_matrix(i, j);
                if (tmp > max_distance) {
                    cost_matrix(i, j) = max_distance + EPSILON;
                }
            }
        }

        Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = hungarian::solve(cost_matrix);
        for (size_t col = 0; col < detection_indices.size(); col++) {
            bool flag = false;
            for (int i = 0; i < indices.rows(); i++)
                if (indices(i, 1) == col) {
                    flag = true;
                    break;
                }
            if (!flag) {
                res.unmatched_detections.push_back(detection_indices[col]);
            }
        }
        for (size_t row = 0; row < track_indices.size(); row++) {
            bool flag = false;
            for (int i = 0; i < indices.rows(); i++)
                if (indices(i, 0) == row){
                    flag = true;
                    break;
                }
            if (!flag) {
                res.unmatched_tracks.push_back(track_indices[row]);
            }
        }

        for (int i = 0; i < indices.rows(); i++) {
            int row = static_cast<int>(indices(i, 0));
            int col = static_cast<int>(indices(i, 1));

            int track_idx = track_indices[row];
            int detection_idx = detection_indices[col];
            if (cost_matrix(row, col) > max_distance) {
                res.unmatched_tracks.push_back(track_idx);
                res.unmatched_detections.push_back(detection_idx);
            } else {
                res.matches.push_back(std::make_pair(track_idx, detection_idx));
            }
        }
        return res;
    }

    CostMatrixType gate_cost_matrix(
        KalmanFilter* kf,
        CostMatrixType &costMat,
        std::vector<Track>& tracks,
        const Detections &detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices,
        float gated_cost,
        bool only_position)
    {
        int gating_dim = (only_position ? 2 : 4);
        double gating_threshold = chi2inv95[gating_dim];
        std::vector<DetectionBox> measurements;
        for (int i : detection_indices) {
            measurements.push_back(detections[i].to_xyah());
        }

        for (size_t i = 0; i < track_indices.size(); i++) {
            Track& track = tracks[track_indices[i]];
            Eigen::Matrix<float, 1, -1> gating_distance = kf->gating_distance(track.mean, track.covariance, measurements, only_position);
            for (int j = 0; j < gating_distance.cols(); j++) {
                if (gating_distance(0, j) > gating_threshold) {
                    costMat(i, j) = gated_cost;
                }
            }
        }
        return costMat;
    }
}