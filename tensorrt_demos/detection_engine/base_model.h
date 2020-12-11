#pragma once

#include "decl_spec.h"
#include "common/datatypes.h"
#include <opencv2/core.hpp>

namespace detector {

    class ENGINE_DECL BaseDetector {
    public:
        virtual ~BaseDetector();
        virtual common::datatypes::DetectionResults inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold) = 0;
    };
}