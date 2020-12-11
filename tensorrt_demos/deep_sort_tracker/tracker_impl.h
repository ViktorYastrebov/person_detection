#pragma once

#include "nn_matching.h"
#include "kalman_filter.h"
#include "base_tracker.h"

namespace deep_sort {

    class TrackerImpl {
    public:
        using DetectionBoxes = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;

        TrackerImpl(float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3,
                const int max_time_since_update = 3, const int max_artificial_updates = 30);

        virtual ~TrackerImpl();
        virtual void predict();
        virtual void update(const common::datatypes::Detections& detections);
        virtual const std::vector<AbstractTracker::TrackPtr> &getTracks() const;

        typedef common::datatypes::CostMatrixType(TrackerImpl::* MetricFunction)(
            std::vector<AbstractTracker::TrackPtr>& tracks,
            const common::datatypes::Detections& detections,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);

        common::datatypes::CostMatrixType gated_metric(
            std::vector<AbstractTracker::TrackPtr>& tracks,
            const common::datatypes::Detections& detections,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices
        );

        common::datatypes::CostMatrixType iou_cost(
            std::vector<AbstractTracker::TrackPtr>& tracks,
            const common::datatypes::Detections& detections,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
        Eigen::VectorXf iou(common::datatypes::DetectionBox& bbox, DetectionBoxes &candidates);

    protected:
        AbstractTracker::TrackPtr initialize_track(const common::datatypes::KalmanMeanMatType &mean, const common::datatypes::KalmanCovAMatType &covariance,
                                  int track_id, int n_init, int max_age, const common::datatypes::Feature &feature, const int class_id);

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

        int frame_counter_;

    protected:
        std::vector<AbstractTracker::TrackPtr> tracks;
        std::function<AbstractTracker::TrackPtr(const common::datatypes::KalmanMeanMatType &mean,
            const common::datatypes::KalmanCovAMatType &covariance,
            int track_id, int n_init, int max_age, const common::datatypes::Feature &feature, const int class_id)> build_func_;

        void reduce_tracks_by_iou();

        int max_time_since_update_;
        int max_artificial_updates_;

    };
}
