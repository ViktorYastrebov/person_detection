#pragma once

#include <vector>
#include <Eigen/Core>


constexpr const float INFTY_COST = 1e5;
constexpr const int FEATURES_SIZE = 512;

using MatchData = std::pair<int, int>;

struct TrackerMatch {
    std::vector<MatchData> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
};

//INFO: DYNAMICM
using CostMatrixType = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;

//typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
using KalmanMeanMatType = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
using KalmanCovAMatType = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;

//typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
//typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KalmanHMeanType = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using KalmanHCovType = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

//TODO: add constexpr to 512 do not hardcode the size
using Feature = Eigen::Matrix<float, 1, FEATURES_SIZE, Eigen::RowMajor>;
using Features = Eigen::Matrix<float, Eigen::Dynamic, FEATURES_SIZE, Eigen::RowMajor>;

