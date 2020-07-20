#pragma once

#include "common.h"
#include "base_model.h"
#include "track.h"


class NearNeighborDisMetric;

class Tracker {
public:

    using DetectionBoxes = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;

    std::unique_ptr<NearNeighborDisMetric> metric_processor;
    float max_iou_distance;
    int max_age;
    int n_init;

    std::unique_ptr<KalmanFilter> kalman_filter;
    int _next_idx;

    std::vector<Track> tracks;
    Tracker(/*NearNeighborDisMetric* metric,*/
        float max_cosine_distance, int nn_budget,
        float max_iou_distance = 0.7,
        int max_age = 30, int n_init = 3);
    void predict();
    void update(const std::vector<DetectionResult>& detections);

    CostMatrixType gated_metric(
        std::vector<Track>& tracks,
        const std::vector<DetectionResult>& dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices
    );

    CostMatrixType iou_cost(
        std::vector<Track>& tracks,
        const std::vector<DetectionResult>& dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);

    Eigen::VectorXf iou(DetectionResult& bbox, DetectionBoxes &candidates);

private:
    void match(const const std::vector<DetectionResult>& detections, TrackerMatch& res);
    void initiate_track(const Track::DetectionRow& detection);
};