#pragma once
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace ganz_camera {

    struct AlarmData {

        bool fromJsonData(const char *ptr);

        std::string time;
        std::vector<cv::Point2i> sn_points;
        std::vector<cv::Rect2i> alarm_areas;
    };
}