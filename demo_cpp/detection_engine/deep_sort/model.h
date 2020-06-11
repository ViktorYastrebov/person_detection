#pragma once

#include "decl_spec.h"
#include "device_utils.h"

#include <opencv2/dnn/dnn.hpp>
#include <string>

namespace deep_sort {

#pragma warning(push)
#pragma warning(disable: 4251)

    class ENGINE_DECL ImageEncoderModel {
    public:
        ImageEncoderModel(const std::string &base_dir, RUN_ON device);
        ~ImageEncoderModel() = default;
        void encode(cv::Mat frame,const std::vector<cv::Rect> &bboxes);

    private:
        const int feature_dim = 128;
        //image shape : [128, 64, 3]
        //const std::pair<int, int> patch_shape = std::make_pair<int, int>(128, 64);
        const double ASPECT = 64.0 / 128.0;
        cv::Mat extract_image_patch(cv::Mat frame, const cv::Rect &bbox);
    private:
        cv::dnn::Net net_;
        std::vector<cv::String> output_layers_;
    };
#pragma warning(pop)
}
