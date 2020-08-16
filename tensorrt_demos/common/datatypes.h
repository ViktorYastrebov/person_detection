#pragma once

#include <vector>
#include <Eigen/Core>

#include "decl_spec.h"

namespace common {
    namespace datatypes {

        constexpr const float INFTY_COST = 1e5;
        //INFO: must be the same size as DeepSort model output
        constexpr const int FEATURES_SIZE = 512;

        using MatchData = std::pair<int, int>;

        struct TrackerMatch {
            std::vector<MatchData> matches;
            std::vector<int> unmatched_tracks;
            std::vector<int> unmatched_detections;
        };
        using CostMatrixType = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
        using KalmanMeanMatType = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
        using KalmanCovAMatType = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;
        using KalmanHMeanType = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
        using KalmanHCovType = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
        using Feature = Eigen::Matrix<float, 1, FEATURES_SIZE, Eigen::RowMajor>;
        using Features = Eigen::Matrix<float, Eigen::Dynamic, FEATURES_SIZE, Eigen::RowMajor>;
        using TrackerResult = std::pair<int, Features>;
        using DetectionBox = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;

        struct COMMON_EXPORT DetectionResult {
            DetectionBox bbox;
            int class_id;
        };
        using DetectionResults = std::vector<DetectionResult>;

        struct COMMON_EXPORT Detection {
            DetectionBox tlwh;
            Feature feature;
            int class_id;
            //TODO: check the usage. May be need to do it for whole matrix
            DetectionBox to_xyah() const;
            DetectionBox to_tlbr() const;
        };
        using Detections = std::vector<Detection>;
    }
}