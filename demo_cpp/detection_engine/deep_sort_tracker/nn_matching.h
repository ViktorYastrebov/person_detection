#pragma once

#include "common.h"
#include <memory>
#include <map>

// TODO: refactor to compiletime or runtime strategy pattern
class NearNeighborDisMetric {
public:
    enum METRIC_TYPE {
        euclidean = 1,
        cosine
    };
    //INFO: might need dynamic size( 512 ) 
    //      remove hardcode
    using TrackerFeatures = std::pair<int, Features>;

    NearNeighborDisMetric(METRIC_TYPE metric, float matching_threshold, int budget);

    CostMatrixType distance(const Features& features, const std::vector<int> &targets);
    void partial_fit(std::vector<TrackerFeatures>& tid_feats, std::vector<int>& active_targets);
    
    float getMatingThreshold() const;

private:

    struct BaseMetricProcessor {
        virtual ~BaseMetricProcessor() = 0;
        virtual Eigen::VectorXf process(const Features& x, const Features& y) const = 0;
    };

    struct CosineMetricDistance : public BaseMetricProcessor {
        ~CosineMetricDistance() = default;
        virtual Eigen::VectorXf process(const Features& x, const Features& y) const;
    };

    struct EuclidianMetricDistance : public BaseMetricProcessor {
        ~EuclidianMetricDistance() = default;
        virtual Eigen::VectorXf process(const Features& x, const Features& y) const;
    private:
        Eigen::MatrixXf preprocess(const Features& x, const Features& y) const;
    };

    //typedef Eigen::VectorXf(NearNeighborDisMetric::*PTRFUN)(const Features&, const Features&);
    //Eigen::VectorXf _nncosine_distance(const Features& x, const Features& y);
    //Eigen::VectorXf _nneuclidean_distance(const Features& x, const Features& y);

    //Eigen::MatrixXf _pdist(const Features& x, const Features& y);
    //Eigen::MatrixXf _cosine_distance(const Features & a, const Features& b, bool data_is_normalized = false);
//private:
//    PTRFUN _metric;
    float mating_threshold_;
    int budget_;
    using SamplesMapType = std::map<int, Features>;
    SamplesMapType samples;
    std::unique_ptr<BaseMetricProcessor> metric_processor_;
};
