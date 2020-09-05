#pragma once

#include "common/datatypes.h"
#include "track.h"
#include "nn_matching.h"

namespace deep_sort {


#pragma warning(push)
#pragma warning(disable: 4251)
    class DEEP_SORT_TRACKER Tracker {
    public:
        using DetectionBoxes = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;
        using TrackPtr = std::shared_ptr<Track>;

        Tracker(float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3);
        virtual ~Tracker() = default;
        virtual void predict();
        virtual void update(const common::datatypes::Detections& detections);

        typedef common::datatypes::CostMatrixType(Tracker::* MetricFunction)(
            std::vector<TrackPtr>& tracks,
            const common::datatypes::Detections& detections,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);

        common::datatypes::CostMatrixType gated_metric(
            std::vector<TrackPtr>& tracks,
            const common::datatypes::Detections& detections,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices
        );

        common::datatypes::CostMatrixType iou_cost(
            std::vector<TrackPtr>& tracks,
            const common::datatypes::Detections& detections,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
        Eigen::VectorXf iou(common::datatypes::DetectionBox& bbox, DetectionBoxes &candidates);

        const std::vector<TrackPtr> &getTracks() const;
    protected:
        TrackPtr initialize_track(const common::datatypes::KalmanMeanMatType &mean, const common::datatypes::KalmanCovAMatType &covariance, int track_id, int n_init, int max_age, const common::datatypes::Feature &feature, const int class_id);

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
    protected:
        std::vector<TrackPtr> tracks;
        std::function<TrackPtr(const common::datatypes::KalmanMeanMatType &mean,
            const common::datatypes::KalmanCovAMatType &covariance,
            int track_id, int n_init, int max_age, const common::datatypes::Feature &feature, const int class_id)> build_func_;
    };
#pragma warning(pop)
}
