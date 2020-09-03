#pragma once

#include "base_model.h"
#include <filesystem>

namespace detector {

    class ENGINE_DECL YoloV5Model : public CommonDetector {
    public:
        YoloV5Model(const std::filesystem::path &model_path, const std::vector<int> &classes_ids, const int BATCH_SIZE = 1);
        ~YoloV5Model() = default;
        common::datatypes::DetectionResults inference(const cv::Mat &imageRGB, const float confidence = 0.5, const float nms_threshold = 0.5) override;
    private:
        cv::Mat preprocessImage(const cv::Mat &imageRGB);
        void prepareBuffer(cv::Mat &prepared);
        common::datatypes::DetectionResults processResults(const cv::Mat &prepared, const float conf, const float nms_thresh);
    private:
        std::vector<int> classes_ids_;
    };
}