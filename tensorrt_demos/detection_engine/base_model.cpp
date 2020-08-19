#include "base_model.h"

namespace detector {
    BaseDetector::~BaseDetector()
    {}

    common::datatypes::DetectionResults CommonDetector::inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold) {
        return {};
    }
}