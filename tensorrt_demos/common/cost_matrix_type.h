#pragma once

#include <Eigen/Core>

namespace common {
    namespace datatypes {
        using CostMatrixType = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
    }
}