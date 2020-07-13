#include "deep_sort.h"


DeepSortModel::DeepSortModel(const std::string &model_path, RUN_ON device)
{
    net_ = cv::dnn::readNetFromONNX(model_path);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
    output_layers_ = net_.getUnconnectedOutLayersNames();
}


void DeepSortModel::test_output(cv::Mat frame, const std::vector<DetectionResult> &detections) {

}