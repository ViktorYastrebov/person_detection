#pragma once

//#include "common.h"
#include "common/datatypes.h"
//#include "base_model.h"
#include "track.h"
#include "nn_matching.h"

//class NearNeighborDisMetric;

class DEEP_SORT_TRACKER Tracker {
public:
    using DetectionBoxes = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;

    Tracker(float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3);
    Tracker() = default;
    void predict();
    void update(const common::datatypes::Detections& detections);

    typedef common::datatypes::CostMatrixType(Tracker::* MetricFunction)(
        std::vector<Track>& tracks,
        const common::datatypes::Detections& detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);

    common::datatypes::CostMatrixType gated_metric(
        std::vector<Track>& tracks,
        const common::datatypes::Detections& detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices
    );

    common::datatypes::CostMatrixType iou_cost(
        std::vector<Track>& tracks,
        const common::datatypes::Detections& detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(common::datatypes::DetectionBox& bbox, DetectionBoxes &candidates);

    const std::vector<Track> &getTracks() const;

private:
    void match(const common::datatypes::Detections& detections, common::datatypes::TrackerMatch& res);
    void initiate_track(const common::datatypes::Detection& detection);
private:
    std::unique_ptr<NearNeighborDisMetric> metric_processor;
    float max_iou_distance;
    int max_age;
    int n_init;
    int _next_idx;
    std::unique_ptr<KalmanFilter> kalman_filter;
    std::vector<Track> tracks;
};
