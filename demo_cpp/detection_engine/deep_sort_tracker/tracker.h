#pragma once

#include "common.h"
#include "base_model.h"
#include "track.h"
#include "nn_matching.h"

//class NearNeighborDisMetric;

class DEEP_SORT_TRACKER Tracker {
public:
    using DetectionBoxes = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;

    Tracker(float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3);
    Tracker() = default;
    void predict();
    void update(const Detections& detections);

    typedef CostMatrixType(Tracker::* MetricFunction)(
        std::vector<Track>& tracks,
        const Detections& detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);

    CostMatrixType gated_metric(
        std::vector<Track>& tracks,
        const Detections& detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices
    );

    CostMatrixType iou_cost(
        std::vector<Track>& tracks,
        const Detections& detections,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(DetectionBox& bbox, DetectionBoxes &candidates);

    const std::vector<Track> &getTracks() const;

private:
    void match(const Detections& detections, TrackerMatch& res);
    void initiate_track(const Detection& detection);
private:
    std::unique_ptr<NearNeighborDisMetric> metric_processor;
    float max_iou_distance;
    int max_age;
    int n_init;
    int _next_idx;
    std::unique_ptr<KalmanFilter> kalman_filter;
    std::vector<Track> tracks;
};
