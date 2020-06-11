#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "decl_spec.h"

class ENGINE_DECL BaseModel {
public:
    virtual ~BaseModel() = 0;
    virtual std::vector<cv::Rect> process(const cv::Mat &frame) = 0;
};