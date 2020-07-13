#pragma once

#include "decl_spec.h"
#include <opencv2/dnn.hpp>

#include "base_model.h"
#include "device_utils.h"

class TRACKER_ENGINE DeepSortModel final {
public:
    DeepSortModel(const std::string &model_path, RUN_ON device = RUN_ON::CPU);
    ~DeepSortModel() = default;

    void test_output(cv::Mat frame, const std::vector<DetectionResult> &detections);

private:
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
};