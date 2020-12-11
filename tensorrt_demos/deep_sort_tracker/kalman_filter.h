#pragma once

#include "deep_sort_types.h"

class KalmanFilter
{
public:
    struct MatResult {
        common::datatypes::KalmanMeanMatType mean;
        common::datatypes::KalmanCovAMatType covariance;
    };

    struct HMatResult {
        common::datatypes::KalmanHMeanType hmean;
        common::datatypes::KalmanHCovType hconv;
    };

    KalmanFilter();
    MatResult initiate(const common::datatypes::DetectionBox& measurement);

    void predict(common::datatypes::KalmanMeanMatType& mean, common::datatypes::KalmanCovAMatType& covariance);
    HMatResult project(const common::datatypes::KalmanMeanMatType& mean, const common::datatypes::KalmanCovAMatType& covariance);
    MatResult update(const common::datatypes::KalmanMeanMatType& mean, const common::datatypes::KalmanCovAMatType& covariance, const common::datatypes::DetectionBox& measurement);
    Eigen::Matrix<float, 1, -1> gating_distance(const common::datatypes::KalmanMeanMatType& mean, const common::datatypes::KalmanCovAMatType& covariance, const std::vector<common::datatypes::DetectionBox>& measurements, bool only_position = false);

private:
    Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
    float _std_weight_position;
    float _std_weight_velocity;
};