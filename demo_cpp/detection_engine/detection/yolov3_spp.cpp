#include "yolov3_spp.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

YoloV3SPP::YoloV3SPP(const std::string &model, const std::vector<int> &classes, const float confidence, RUN_ON device)
    :conf_threshold_(confidence)
    , filtered_classes_(classes)
{
    net_ = cv::dnn::readNetFromONNX(model);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
    output_layers_ = net_.getUnconnectedOutLayersNames();
}

//INFO: output is different from the YoloV3
//     3 dims with:
//     1x25200x85, 1 looks like a batch size

std::vector<DetectionResult> YoloV3SPP::process(const cv::Mat &frame) {
    constexpr const double NORM_FACTOR = 1.0 / 255.0;
    constexpr const float INPUT_SIZE = 640;

    auto width = static_cast<float>(frame.size().width);
    auto height = static_cast<float>(frame.size().height);
    auto ratio = std::min(INPUT_SIZE / width, INPUT_SIZE / height);

    auto new_width = static_cast<int>(std::round(width * ratio));
    auto new_height = static_cast<int>(std::round(height *ratio));

    //dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height), 0.0, 0.0, cv::INTER_LINEAR);
    auto width_pad = static_cast<int>(INPUT_SIZE) - new_width;
    auto height_pad = static_cast<int>(INPUT_SIZE) - new_height;
    
    cv::Mat input(cv::Size(640, 640), resized.type());
    cv::copyMakeBorder(resized, input,0, height_pad, 0, width_pad, cv::BORDER_CONSTANT, cv::Scalar(114,114, 144));

    cv::imwrite("prepared_input.png", input);

    auto blob = cv::dnn::blobFromImage(input, NORM_FACTOR, cv::Size(), cv::Scalar(), true, false);
    net_.setInput(blob);

    std::vector<cv::Mat> ret;
    net_.forward(ret, output_layers_);

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classes;

    for (const auto &mat : ret) {
        int planes = mat.size[0];
        int rows = mat.size[1];
        int cols = mat.size[2];
        for (int i = 0; i < planes; ++i) {
            // assume there is only 1 plane
            cv::Mat slice(rows, cols, CV_32FC1, mat.data + mat.step[0] * i);
            //auto width = frame.size().width;
            //auto height = frame.size().height;

            for (int row_it = 0; row_it < slice.rows; ++row_it) {
                const float * row_ptr = slice.ptr<float>(row_it);
                auto value_it = std::max_element(&row_ptr[5], &row_ptr[slice.cols]);
                auto value = *value_it;
                std::size_t class_id = std::distance(&row_ptr[5], value_it);

                auto it = std::find(filtered_classes_.cbegin(), filtered_classes_.cend(), class_id);
                if (it != filtered_classes_.cend() && value > conf_threshold_) {
                    int center_x = static_cast<int>(row_ptr[0] * INPUT_SIZE);
                    int center_y = static_cast<int>(row_ptr[1] * INPUT_SIZE);
                    int w = static_cast<int>(row_ptr[2] * INPUT_SIZE);
                    int h = static_cast<int>(row_ptr[3] * INPUT_SIZE);
                    int x = static_cast<int>(center_x - w / 2);
                    int y = static_cast<int>(center_y - h / 2);
                    bboxes.push_back(cv::Rect(x, y, w, h));
                    scores.push_back(value);
                    classes.push_back(class_id);
                }
            }
        }
    }
    std::vector<DetectionResult> output;
    std::vector<int> idxs;
    cv::dnn::NMSBoxes(bboxes, scores, 0.5f, 0.4f, idxs);
    for (const auto &idx : idxs) {
        output.push_back({ bboxes[idx], classes[idx] });
        cv::rectangle(input, bboxes[idx], cv::Scalar(0, 0, 255));
    }
    cv::imwrite("processed.png", input);

    return output;
}