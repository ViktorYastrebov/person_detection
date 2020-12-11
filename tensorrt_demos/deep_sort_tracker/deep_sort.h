#pragma once

#include "decl_spec.h"

#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "deep_sort_types.h"

namespace deep_sort {
    class DEEP_SORT_TRACKER DeepSort final {
    public:
        static constexpr const int MAX_BATCH_SIZE = 32;
        static constexpr const int INPUT_W = 64;
        static constexpr const int INPUT_H = 128;
        static constexpr const int OUTPUT_SIZE = 512;

        DeepSort(const std::filesystem::path &model_path, const int BATCH_SIZE = MAX_BATCH_SIZE);
        ~DeepSort();
        common::datatypes::Detections getFeatures(const cv::Mat &imageRGB, const common::datatypes::DetectionResults &bboxes);
    private:
        struct Pimpl;
        Pimpl *pimpl_;
    };
}