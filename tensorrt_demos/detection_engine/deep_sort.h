#pragma once

#include "decl_spec.h"

#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include <cuda_runtime_api.h>
#include "common/logging.h"
#include "common/common.h"
#include "deep_sort_tracker/deep_sort_types.h"


namespace deep_sort_tracker {
    class ENGINE_DECL DeepSort final {
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