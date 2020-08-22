#pragma once
#include "common/decl_spec.h"
#include "common/datatypes.h"

namespace hungarian {
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> COMMON_EXPORT solve(const common::datatypes::CostMatrixType &cost_matrix);
}