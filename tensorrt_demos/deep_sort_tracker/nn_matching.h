#pragma once

#include "common/cost_matrix_type.h"
#include "deep_sort_types.h"
#include <memory>
#include <map>

// TODO: refactor to compiletime or runtime strategy pattern
class NearNeighborDisMetric {
public:
    enum METRIC_TYPE {
        euclidean = 1,
        cosine
    };
    using TrackerFeatures = std::pair<int, common::datatypes::Features>;

    NearNeighborDisMetric(METRIC_TYPE metric, float matching_threshold, int budget);

    common::datatypes::CostMatrixType distance(const common::datatypes::Features& features, const std::vector<int> &targets);
    void partial_fit(std::vector<TrackerFeatures>& tid_feats, std::vector<int>& active_targets);
    
    float getMatingThreshold() const;

private:

    struct BaseMetricProcessor {
        virtual ~BaseMetricProcessor() = 0;
        virtual Eigen::VectorXf process(const common::datatypes::Features& x, const common::datatypes::Features& y) const = 0;
    };

    struct CosineMetricDistance : public BaseMetricProcessor {
        ~CosineMetricDistance() = default;
        virtual Eigen::VectorXf process(const common::datatypes::Features& x, const common::datatypes::Features& y) const;
    };

    struct EuclidianMetricDistance : public BaseMetricProcessor {
        ~EuclidianMetricDistance() = default;
        virtual Eigen::VectorXf process(const common::datatypes::Features& x, const common::datatypes::Features& y) const;
    private:
        Eigen::MatrixXf preprocess(const common::datatypes::Features& x, const common::datatypes::Features& y) const;
    };
    float mating_threshold_;
    int budget_;
    using SamplesMapType = std::map<int, common::datatypes::Features>;
    SamplesMapType samples;
    std::unique_ptr<BaseMetricProcessor> metric_processor_;
};
