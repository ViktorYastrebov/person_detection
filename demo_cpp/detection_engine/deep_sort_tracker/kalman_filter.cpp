#include "kalman_filter.h"
#include <Eigen/Cholesky>

using MatDetection = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using MatAllDetections  = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;

KalmanFilter::KalmanFilter()
{
    constexpr const int ndim = 4;
    constexpr const double dt = 1.0;

    _motion_mat = Eigen::MatrixXf::Identity(8, 8);
    for (int i = 0; i < ndim; i++) {
        _motion_mat(i, ndim + i) = dt;
    }
    _update_mat = Eigen::MatrixXf::Identity(4, 8);
    _std_weight_position = 1.0 / 20;
    _std_weight_velocity = 1.0 / 160;
}

KalmanFilter::MatResult KalmanFilter::initiate(const DetectionResult& measurement) {
    MatDetection mean_pos({ measurement.bbox.x, measurement.bbox.y, measurement.bbox.width, measurement.bbox.height });
    MatDetection mean_vel({ 0,0,0,0 });
    //for (int i = 0; i < 4; i++) {
    //    mean_vel(i) = 0;
    //}

    //KAL_MEAN mean;
    KalmanMeanMatType mean;
    for (int i = 0; i < 8; i++) {
        if (i < 4) {
            mean(i) = mean_pos(i);
        } else {
            mean(i) = mean_vel(i - 4);
        }
    }

    //KAL_MEAN std;
    KalmanMeanMatType std;
    //std(0) = 2 * _std_weight_position * measurement[3];
    //std(1) = 2 * _std_weight_position * measurement[3];
    //std(2) = 1e-2;
    //std(3) = 2 * _std_weight_position * measurement[3];
    //std(4) = 10 * _std_weight_velocity * measurement[3];
    //std(5) = 10 * _std_weight_velocity * measurement[3];
    //std(6) = 1e-5;
    //std(7) = 10 * _std_weight_velocity * measurement[3];

    std(0) = 2 * _std_weight_position * measurement.bbox.width;
    std(1) = 2 * _std_weight_position * measurement.bbox.width;
    std(2) = 1e-2;
    std(3) = 2 * _std_weight_position * measurement.bbox.width;
    std(4) = 10 * _std_weight_velocity * measurement.bbox.width;
    std(5) = 10 * _std_weight_velocity * measurement.bbox.width;
    std(6) = 1e-5;
    std(7) = 10 * _std_weight_velocity * measurement.bbox.width;

    KalmanMeanMatType tmp = std.array().square();
    KalmanCovAMatType var = tmp.asDiagonal();
    return { mean, var };
}

void KalmanFilter::predict(KalmanMeanMatType& mean, KalmanCovAMatType& covariance) {
    MatDetection std_pos;
    std_pos << _std_weight_position * mean(3),
        _std_weight_position * mean(3),
        1e-2,
        _std_weight_position * mean(3);
    MatDetection std_vel;
    std_vel << _std_weight_velocity * mean(3),
        _std_weight_velocity * mean(3),
        1e-5,
        _std_weight_velocity * mean(3);
    //KAL_MEAN tmp;
    KalmanMeanMatType tmp;
    tmp.block<1, 4>(0, 0) = std_pos;
    tmp.block<1, 4>(0, 4) = std_vel;
    tmp = tmp.array().square();
    //KAL_COVA motion_cov = tmp.asDiagonal();
    //KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
    //KAL_COVA covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
    KalmanCovAMatType motion_cov = tmp.asDiagonal();
    KalmanMeanMatType mean1 = _motion_mat * mean.transpose();
    KalmanCovAMatType covariance1 = _motion_mat * covariance * (_motion_mat.transpose());
    covariance1 += motion_cov;

    mean = mean1;
    covariance = covariance1;
}

KalmanFilter::HMatResult KalmanFilter::project(const KalmanMeanMatType& mean, const KalmanCovAMatType& covariance) {
    MatDetection std;
    std << _std_weight_position * mean(3), _std_weight_position * mean(3),
        1e-1, _std_weight_position * mean(3);
    //KAL_HMEAN mean1 = _update_mat * mean.transpose();
    KalmanHMeanType mean1 = _update_mat * mean.transpose();
    //KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
    KalmanHCovType covariance1 = _update_mat * covariance * (_update_mat.transpose());
    Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
    diag = diag.array().square().matrix();
    covariance1 += diag;
    //    covariance1.diagonal() << diag;
    return { mean1, covariance1 };
}

KalmanFilter::MatResult KalmanFilter::update(const KalmanMeanMatType& mean, const KalmanCovAMatType& covariance, const DetectionResult& measurement) {
    auto pa = project(mean, covariance);
    KalmanHMeanType projected_mean = pa.hmean;
    KalmanHCovType projected_cov = pa.hconv;

    //chol_factor, lower =
    //scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
    //kalmain_gain =
    //scipy.linalg.cho_solve((cho_factor, lower),
    //np.dot(covariance, self._upadte_mat.T).T,
    //check_finite=False).T
    MatDetection measure_mat({ measurement.bbox.x, measurement.bbox.y, measurement.bbox.width, measurement.bbox.height });

    Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
    Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
    Eigen::Matrix<float, 1, 4> innovation = measure_mat - projected_mean; //eg.1x4
    auto tmp = innovation * (kalman_gain.transpose());
    KalmanMeanMatType new_mean = (mean.array() + tmp.array()).matrix();
    KalmanCovAMatType new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
    return { new_mean, new_covariance };
}

Eigen::Matrix<float, 1, -1> KalmanFilter::gating_distance(const KalmanMeanMatType& mean, const KalmanCovAMatType& covariance, const std::vector<DetectionResult>& measurements, bool only_position) {
    auto pa = project(mean, covariance);
    if (only_position) {
        //printf("not implement!");
        //exit(0);
        throw std::runtime_error("Only position is not implemented");
    }
    KalmanHMeanType  mean1 = pa.hmean;
    KalmanHCovType covariance1 = pa.hconv;

    //    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
    //DETECTBOXSS d(measurements.size(), 4);
    MatAllDetections d(measurements.size(), 4);
    int pos = 0;
    //TODO: investigate to use the detections as th EigenMat
    //      cost of refactoring
    for (const auto det : measurements) {
        MatDetection mat_box({ det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height });
        d.row(pos++) = mat_box - mean1;
    }
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
    Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
    auto zz = ((z.array())*(z.array())).matrix();
    auto square_maha = zz.colwise().sum();
    return square_maha;
}
