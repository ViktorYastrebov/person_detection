#pragma once

#include "common.h"
#include "base_model.h"
#include "tracker.h"
#include "track.h"
#include "kalman_filter.h"

class Treck;

namespace linear_assignment {

    TrackerMatch DEEP_SORT_TRACKER matching_cascade(Tracker* distance_metric,
        Tracker::MetricFunction metric_function,
        //tracker::GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        int cascade_depth,
        std::vector<Track>& tracks,
        //const DETECTIONS& detections,
        //const std::vector<DetectionResult> &detections,
        const Detections &detections,
        std::vector<int> &track_indices,
        std::vector<int> detection_indices = std::vector<int>());
    TrackerMatch DEEP_SORT_TRACKER min_cost_matching(
        Tracker* distance_metric,
        //tracker::GATED_METRIC_FUNC distance_metric_func,
        Tracker::MetricFunction metric_function,
        float max_distance,
        std::vector<Track>& tracks,
        //const DETECTIONS& detections,
        //const std::vector<DetectionResult> &detections,
        const Detections &detections,
        std::vector<int>& track_indices,
        std::vector<int>& detection_indices);
    CostMatrixType DEEP_SORT_TRACKER gate_cost_matrix(
        KalmanFilter* kf,
        //DYNAMICM& cost_matrix,
        CostMatrixType &costMat,
        std::vector<Track>& tracks,
        //const DETECTIONS& detections,
        //const std::vector<DetectionResult> &detections,
        const Detections &detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices,
        float gated_cost = INFTY_COST,
        bool only_position = false);
}