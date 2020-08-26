#include "hungarian.h"

namespace hungarian {
    using namespace common::datatypes;

    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> solve(const CostMatrixType &cost_matrix) {
        int rows = cost_matrix.rows();
        int cols = cost_matrix.cols();
        Matrix<double> matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                matrix(row, col) = cost_matrix(row, col);
            }
        }
        //Munkres get matrix;
        Munkres<double> m;
        m.solve(matrix);

        //
        std::vector<std::pair<int, int>> pairs;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int tmp = (int)matrix(row, col);
                if (tmp == 0) pairs.push_back(std::make_pair(row, col));
            }
        }
        //
        int count = pairs.size();
        Eigen::Matrix<float, -1, 2, Eigen::RowMajor> re(count, 2);
        for (int i = 0; i < count; i++) {
            re(i, 0) = pairs[i].first;
            re(i, 1) = pairs[i].second;
        }
        return re;
    }

    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> solve(Matrix<double> &matrix) {
        Munkres<double> m;
        m.solve(matrix);

        std::vector<std::pair<int, int>> pairs;
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.columns(); col++) {
                int tmp = static_cast<int>(matrix(row, col));
                if (tmp == 0) {
                    pairs.push_back(std::make_pair(row, col));
                }
            }
        }
        int count = pairs.size();
        Eigen::Matrix<float, -1, 2, Eigen::RowMajor> re(count, 2);
        for (int i = 0; i < count; i++) {
            re(i, 0) = pairs[i].first;
            re(i, 1) = pairs[i].second;
        }
        return re;
    }
}
