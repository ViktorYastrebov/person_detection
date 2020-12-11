#pragma once

#include <vector>
#include <Eigen/Core>

#include "decl_spec.h"

namespace common {
    namespace datatypes {
        using MatchData = std::pair<int, int>;

        struct TrackerMatch {
            std::vector<MatchData> matches;
            std::vector<int> unmatched_tracks;
            std::vector<int> unmatched_detections;
        };
        using DetectionBox = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;

#pragma warning(push)
#pragma warning(disable: 4251)
        struct COMMON_EXPORT DetectionResult {
            DetectionBox bbox;
            int class_id;
        };
#pragma warning(pop)
        using DetectionResults = std::vector<DetectionResult>;
    }
}