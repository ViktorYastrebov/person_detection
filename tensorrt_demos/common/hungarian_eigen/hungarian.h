#pragma once
#include "common/decl_spec.h"
#include "common/cost_matrix_type.h"
#include "munkres/munkres.h"

namespace hungarian {
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> COMMON_EXPORT solve(const common::datatypes::CostMatrixType &cost_matrix);
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> COMMON_EXPORT solve(Matrix<double> &matrix);
}