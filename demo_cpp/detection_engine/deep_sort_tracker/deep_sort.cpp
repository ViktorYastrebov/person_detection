#include "deep_sort.h"
#include <iostream>


DeepSortModel::DeepSortModel(const std::string &model_path, RUN_ON device)
{
    net_ = cv::dnn::readNetFromONNX(model_path);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
    output_layers_ = net_.getUnconnectedOutLayersNames();
}

std::vector<DeepSortModel::FeaturesType> DeepSortModel::getFeatures(cv::Mat frame, const std::vector<DetectionResult> &detections) {

    constexpr const double NORM_FACTOR = 1.0 / 255.0;
    constexpr const int BATCH = 4;

    std::vector<FeaturesType> features;

    int num = static_cast<int>(detections.size()) / BATCH;
    int rest = static_cast<int>(detections.size()) % BATCH;

    for (int i = 0; i < num; ++i) {
        std::vector<cv::Mat> frames;

        for (int j = 0; j < BATCH; ++j) {
            auto bbox = detections[i*BATCH + j].bbox;
            bbox.x = std::max(bbox.x, 0);
            bbox.y = std::max(bbox.y, 0);
            frames.push_back(frame(bbox));
        }

        auto blob = cv::dnn::blobFromImages(frames, NORM_FACTOR, cv::Size(64, 128), cv::Scalar(0, 0, 0), true, false);
        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, output_layers_);

        for (auto &mat : outputs) {
            for (int rowId = 0; rowId < mat.rows; ++rowId) {
                const float *ptr = mat.ptr<float>(rowId);
                FeaturesType feature(&ptr[0], &ptr[mat.cols]);
                features.push_back(feature);
            }
        }
    }

    if (rest) {
        std::vector<cv::Mat> frames;
        for (int i = num * BATCH; i < detections.size(); ++i) {
            auto bbox = detections[i].bbox;
            bbox.x = std::max(bbox.x, 0);
            bbox.y = std::max(bbox.y, 0);
            frames.push_back(frame(bbox));
        }
        auto blob = cv::dnn::blobFromImages(frames, NORM_FACTOR, cv::Size(64, 128), cv::Scalar(0, 0, 0), true, false);
        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, output_layers_);

        // INFO: rest of the, Mat contains ROW = BATCH so we do not need all of them
        for (auto &mat : outputs) {
            for (int rowId = 0; rowId < rest; ++rowId) {
                 const float *ptr = mat.ptr<float>(rowId);
                 std::vector<float> feature(&ptr[0], &ptr[mat.cols]);
                 features.push_back(feature);
            }
        }
    }
    return features;
}