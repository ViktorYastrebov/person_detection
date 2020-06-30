#pragma once

#include <opencv2/dnn/dnn.hpp>

#include <vector>

#include "base_model.h"
#include "decl_spec.h"
#include "device_utils.h"

#pragma warning(push)
#pragma warning(disable: 4251)

class ENGINE_DECL YoloV3Batched : public BaseModelBatched {
public:
    YoloV3Batched(const std::string &model, const std::string &config, const std::vector<int> &classes, const float confidence = 0.3, RUN_ON device = RUN_ON::CPU);
    ~YoloV3Batched() = default;
    std::vector<std::vector<DetectionResult>> process(const std::vector<cv::Mat> &frames);
private:
    //INFO: use default image size (320, 320), possible values are: 416, 320, depends on cfg file
    const int INPUT_SIZE = 320;
    const int BATCH_SIZE = 64;
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
    const float conf_threshold_;
    std::vector<int> filtered_classes_;
};

#pragma warning(pop)