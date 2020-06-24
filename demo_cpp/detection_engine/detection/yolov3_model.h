#pragma once

#include <opencv2/dnn/dnn.hpp>

#include <vector>

#include "base_model.h"
#include "decl_spec.h"
#include "device_utils.h"

#pragma warning(push)
#pragma warning(disable: 4251)

class ENGINE_DECL YoloV3: public BaseModel {
public:
    YoloV3(const std::string &model, const std::string &config, const std::vector<int> &classes, const float confidence = 0.3, RUN_ON device = RUN_ON::CPU);
    ~YoloV3() = default;
    std::vector<DetectionResult> process(const cv::Mat &frame);
private:
    //INFO: use default image size (320, 320), possible values are: 416, 320, depends on cfg file
    const int INPUT_SIZE = 320;
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
    const float conf_threshold_;
    std::vector<int> filtered_classes_;
};

#pragma warning(pop)