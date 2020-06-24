#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "decl_spec.h"


#pragma warning(push)
#pragma warning(disable: 4251)

struct ENGINE_DECL DetectionResult {
    cv::Rect bbox;
    int class_id;
};

class ENGINE_DECL BaseModel {
public:
    virtual ~BaseModel() = 0;
    virtual std::vector<DetectionResult> process(const cv::Mat &frame) = 0;
};

#pragma warning(pop)