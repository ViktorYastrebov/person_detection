//#pragma once
//#include "munkres/munkres.h"
//#include "munkres/adapters/boostmatrixadapter.h"
//#include "../feature/dataType.h"
//
//class Hungarian {
//public:
//    static Eigen::Matrix<float, -1, 2, Eigen::RowMajor> Solve(const DYNAMICM &cost_matrix);
//};
#pragma once
#include "common.h"

namespace hungarian {
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> solve(const CostMatrixType &cost_matrix);
}