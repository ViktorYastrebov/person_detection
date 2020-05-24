#pragma once

#include <vector>

namespace ganz_camera {

    struct FaceData {
        int x;
        int y;
        int width;
        int height;
        float confidence;
        double temperature;
    };

    struct FaceDataVector {
        bool fromJsonData(const char *data);
        std::vector<FaceData> faces_data_;
    };

}