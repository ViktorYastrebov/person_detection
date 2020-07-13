#include "yolo_tiny.h"

YoloV3Tiny::YoloV3Tiny(const std::string &path, const std::string &config, const std::vector<int> &classes, const float confidence, RUN_ON device)
    : conf_threshold_(confidence)
    , filtered_classes_(classes)
{
    net_ = cv::dnn::readNet(path, config);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
    output_layers_ = net_.getUnconnectedOutLayersNames();
}

std::vector<DetectionResult> YoloV3Tiny::process(const cv::Mat &frame) {
    constexpr const double NORM_FACTOR = 1.0 / 255.0;

    net_.setInput(cv::dnn::blobFromImage(frame, NORM_FACTOR, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(0, 0, 0), true, false));

    std::vector<cv::Mat> ret;
    net_.forward(ret, output_layers_);

    auto width = frame.size().width;
    auto height = frame.size().height;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;
    for (const auto &mat : ret) {
        for (int i = 0; i < mat.rows; ++i) {
            const float * row = mat.ptr<float>(i);
            auto value = std::max_element(&row[5], &row[mat.cols]);
            std::size_t class_id = std::distance(&row[5], value);
            auto it = std::find(filtered_classes_.cbegin(), filtered_classes_.cend(), class_id);
            if (it != filtered_classes_.cend() && *value > conf_threshold_) {
                int center_x = static_cast<int>(row[0] * width);
                int center_y = static_cast<int>(row[1] * height);
                int w = static_cast<int>(row[2] * width);
                int h = static_cast<int>(row[3] * height);
                int x = static_cast<int>(center_x - w / 2);
                int y = static_cast<int>(center_y - h / 2);
                bboxes.push_back(cv::Rect(x, y, w, h));
                scores.push_back(*value);
                classes.push_back(class_id);
            }
        }
    }

    std::vector<DetectionResult> output;
    std::vector<int> idxs;
    cv::dnn::NMSBoxes(bboxes, scores, 0.5f, 0.4f, idxs);
    for (const auto &idx : idxs) {
        output.push_back({ bboxes[idx], classes[idx] });
    }
    return output;
}
