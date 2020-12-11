#pragma once

#include "common/cost_matrix_type.h"
#include "deep_sort_types.h"
#include "tracker_impl.h"
#include "track.h"
#include "kalman_filter.h"

namespace linear_assignment {
    common::datatypes::TrackerMatch DEEP_SORT_TRACKER matching_cascade(deep_sort::TrackerImpl* distance_metric,
        deep_sort::TrackerImpl::MetricFunction metric_function,
        float max_distance,
        int cascade_depth,
        std::vector<deep_sort::AbstractTracker::TrackPtr>& tracks,
        const common::datatypes::Detections &detections,
        std::vector<int> &track_indices,
        std::vector<int> detection_indices = std::vector<int>());
    common::datatypes::TrackerMatch DEEP_SORT_TRACKER min_cost_matching(
        deep_sort::TrackerImpl* distance_metric,
        deep_sort::TrackerImpl::MetricFunction metric_function,
        float max_distance,
        std::vector<deep_sort::AbstractTracker::TrackPtr>& tracks,
        const common::datatypes::Detections &detections,
        std::vector<int>& track_indices,
        std::vector<int>& detection_indices);
    common::datatypes::CostMatrixType DEEP_SORT_TRACKER gate_cost_matrix(
        KalmanFilter* kf,
        common::datatypes::CostMatrixType &costMat,
        std::vector<deep_sort::AbstractTracker::TrackPtr>& tracks,
        const common::datatypes::Detections &detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices,
        float gated_cost = common::datatypes::INFTY_COST,
        bool only_position = false);
}