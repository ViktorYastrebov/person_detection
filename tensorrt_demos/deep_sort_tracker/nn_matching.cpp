#include "nn_matching.h"

using namespace common::datatypes;

NearNeighborDisMetric::BaseMetricProcessor::~BaseMetricProcessor()
{}

Eigen::VectorXf NearNeighborDisMetric::CosineMetricDistance::process(const Features& x, const Features& y) const {
    Eigen::MatrixXf distances = 1.0 - (x*y.transpose()).array();
    Eigen::VectorXf res = distances.colwise().minCoeff().transpose();
    return res;
}

Eigen::VectorXf NearNeighborDisMetric::EuclidianMetricDistance::process(const Features& x, const Features& y) const {
    Eigen::MatrixXf distances = preprocess(x, y);
    Eigen::VectorXf res = distances.colwise().maxCoeff().transpose();
    res = res.array().max(Eigen::VectorXf::Zero(res.rows()).array());
    return res;
}

Eigen::MatrixXf NearNeighborDisMetric::EuclidianMetricDistance::preprocess(const Features& x, const Features& y) const {
    auto len1 = x.rows();
    auto len2 = y.rows();
    if (len1 == 0 || len2 == 0) {
        return Eigen::MatrixXf::Zero(len1, len2);
    }
    Eigen::MatrixXf res = x * y.transpose()* -2;
    res = res.colwise() + x.rowwise().squaredNorm();
    res = res.rowwise() + y.rowwise().squaredNorm().transpose();
    res = res.array().max(Eigen::MatrixXf::Zero(res.rows(), res.cols()).array());
    return res;
}


NearNeighborDisMetric::NearNeighborDisMetric(METRIC_TYPE metric, float matching_threshold, int budget)
    : mating_threshold_(matching_threshold)
    , budget_(budget)
    , samples()
{
    switch (metric) {
    case METRIC_TYPE::cosine:
    {
        metric_processor_.reset(new CosineMetricDistance());
    }break;
    case METRIC_TYPE::euclidean:
    {
        metric_processor_.reset(new EuclidianMetricDistance());
    }break;
    }
}

CostMatrixType NearNeighborDisMetric::distance(const Features& features, const std::vector<int> &targets) {
    CostMatrixType cost_matrix = Eigen::MatrixXf::Zero(targets.size(), features.rows());
    int idx = 0;
    for (int target : targets) {
        cost_matrix.row(idx) = metric_processor_->process(samples[target], features);
        idx++;
    }
    return cost_matrix;
}


void NearNeighborDisMetric::partial_fit(std::vector<TrackerFeatures>& tid_feats, std::vector<int>& active_targets) {
    //  python code:
    //  let feature(target_id) append to samples && delete not comfirmed target_id from samples.
    //  update samples

    for (TrackerFeatures& data : tid_feats) {
        int track_id = data.first;
        Features newFeatOne = data.second;

        if (samples.find(track_id) != samples.end()) {//append
            auto oldSize = samples[track_id].rows();
            auto addSize = newFeatOne.rows();
            auto newSize = oldSize + addSize;

            if (newSize <= this->budget_) {
                Features newSampleFeatures(newSize, FEATURES_SIZE);
                newSampleFeatures.block(0, 0, oldSize, FEATURES_SIZE) = samples[track_id];
                newSampleFeatures.block(oldSize, 0, addSize, FEATURES_SIZE) = newFeatOne;
                samples[track_id] = newSampleFeatures;
            } else {
                if (oldSize < budget_) {//original space is not enough;
                    Features newSampleFeatures(budget_, FEATURES_SIZE);
                    if (addSize >= budget_) {
                        newSampleFeatures = newFeatOne.block(0, 0, budget_, FEATURES_SIZE);
                    } else {
                        newSampleFeatures.block(0, 0, budget_ - addSize, FEATURES_SIZE) =
                            samples[track_id].block(addSize - 1, 0, budget_ - addSize, FEATURES_SIZE).eval();
                        newSampleFeatures.block(budget_ - addSize, 0, addSize, FEATURES_SIZE) = newFeatOne;
                    }
                    samples[track_id] = newSampleFeatures;
                } else {//original space is ok;
                    if (addSize >= budget_) {
                        samples[track_id] = newFeatOne.block(0, 0, budget_, FEATURES_SIZE);
                    } else {
                        samples[track_id].block(0, 0, budget_ - addSize, FEATURES_SIZE) =
                            samples[track_id].block(addSize - 1, 0, budget_ - addSize, FEATURES_SIZE).eval();
                        samples[track_id].block(budget_ - addSize, 0, addSize, 128) = newFeatOne;
                    }
                }
            }
        } else {//not exit, create new one;
            samples[track_id] = newFeatOne;
        }
    }//add features;

    for (SamplesMapType::iterator i = samples.begin(); i != samples.end();) {
        bool flag = false;
        for (int j : active_targets) {
            if (j == i->first) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            samples.erase(i++);
        } else {
            ++i;
        }
    }
}

float NearNeighborDisMetric::getMatingThreshold() const {
    return mating_threshold_;
}
