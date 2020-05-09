#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <vector>

#include "decl_spec.h"
#include "device_utils.h"


class ENGINE_DECL YoloV3 {
public:
    YoloV3(const std::string &model, const std::string &config, RUN_ON device = RUN_ON::CPU);
    ~YoloV3() = default;
    std::vector<cv::Rect> process(const cv::Mat &frame);
private:
    //INFO: use default image size (320, 320), possible values are: 416, 320, depends on cfg file
    const int INPUT_SIZE = 320;
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
};
