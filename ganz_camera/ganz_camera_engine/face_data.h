#pragma once

#include "decl_spec.h"
#include <vector>

#pragma warning(push)
#pragma warning(disable: 4251)

namespace ganz_camera {

    struct GANZ_CAMERA_ENGINE_DECL FaceData {
        int x;
        int y;
        int width;
        int height;
        float confidence;
        double temperature;
    };

    struct GANZ_CAMERA_ENGINE_DECL FaceDataVector {
        bool fromJsonData(const char *data);
        std::vector<FaceData> faces_data_;
    };

}

#pragma warning(pop)