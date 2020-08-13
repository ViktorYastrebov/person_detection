#pragma once

#include "decl_spec.h"
#include <opencv2/dnn.hpp>

#include "base_model.h"
#include "device_utils.h"
#include "common.h"

#pragma warning(push)
#pragma warning(disable: 4251)

class DEEP_SORT_TRACKER DeepSortModel final {
public:
    //using FeaturesType = std::vector<float>;
    DeepSortModel(const std::string &model_path, RUN_ON device = RUN_ON::CPU);
    ~DeepSortModel() = default;

    Detections getFeatures(cv::Mat frame, const std::vector<DetectionResult> &detections);
private:
    static const int BATCH = 32;
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
};

#pragma warning(pop)