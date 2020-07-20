#pragma once

#include "common.h"
#include "base_model.h"

class KalmanFilter
{
public:

    struct MatResult {
        KalmanMeanMatType mean;
        KalmanCovAMatType covariance;
    };

    struct HMatResult {
        KalmanHMeanType hmean;
        KalmanHCovType hconv;
   };

   KalmanFilter();
   MatResult initiate(const DetectionResult& measurement);

    void predict(KalmanMeanMatType& mean, KalmanCovAMatType& covariance);
    HMatResult project(const KalmanMeanMatType& mean, const KalmanCovAMatType& covariance);
    MatResult update(const KalmanMeanMatType& mean, const KalmanCovAMatType& covariance, const DetectionResult& measurement);
    Eigen::Matrix<float, 1, -1> gating_distance(const KalmanMeanMatType& mean, const KalmanCovAMatType& covariance, const std::vector<DetectionResult>& measurements, bool only_position = false);

private:
    Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
    float _std_weight_position;
    float _std_weight_velocity;
};