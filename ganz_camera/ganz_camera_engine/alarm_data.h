#pragma once

#include "decl_spec.h"
#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable: 4251)

namespace ganz_camera {

    struct GANZ_CAMERA_ENGINE_DECL AlarmData {

        bool fromJsonData(const char *ptr);

        std::string time;
        std::vector<cv::Point2i> sn_points;
        std::vector<cv::Rect2i> alarm_areas;
    };
}

#pragma warning(pop)