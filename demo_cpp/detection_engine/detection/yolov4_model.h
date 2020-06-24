#pragma once

#include <opencv2/dnn/dnn.hpp>
#include "base_model.h"

#include "decl_spec.h"
#include "device_utils.h"

#pragma warning(push)
#pragma warning(disable: 4251)

//DOES NOT WORK DUE TO ACTIVATION LAYER, it's not linear
class ENGINE_DECL YoloV4 : public BaseModel {
public:
    YoloV4(const std::string &model, const std::string &config, const std::vector<int> &classes, const float confidence = 0.3, RUN_ON device = RUN_ON::CPU);
    ~YoloV4() = default;
    std::vector<DetectionResult> process(const cv::Mat &frame);
private:
    const int INPUT_SIZE = 512;
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
    const float conf_threshold_;
    std::vector<int> filtered_classes_;
};

#pragma warning(pop)