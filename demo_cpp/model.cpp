#include <fstream>
#include <opencv2/imgproc.hpp>
#include "model.h"


#include <iostream>

SSDliteMobileV2::SSDliteMobileV2(const std::string &path, const std::string &config, RUN_ON device)
    :BaseModel()
{
    net_ = cv::dnn::readNetFromTensorflow(path, config);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
}

cv::Mat SSDliteMobileV2::process(const cv::Mat &frame) {
    net_.setInput(cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(), true, false));

    cv::Mat ret = net_.forward();
    cv::Mat slice(ret.size[2], ret.size[3], CV_32F, ret.ptr<uchar>(0, 0));
    for (int i = 0; i < slice.rows; ++i) {
        const float * row = slice.ptr<float>(i);
        auto score = row[2];
        if (score > 0.6) {
            auto x = static_cast<int>(row[3] * frame.cols);
            auto y = static_cast<int>(row[4] * frame.rows);
            auto x2 =static_cast<int>(row[5] * frame.cols);
            auto y2 =static_cast<int>(row[6] * frame.cols);
            cv::rectangle(frame, cv::Rect(x,y, x2 - x, y2 - y), cv::Scalar(255, 0, 0));
        }
    }
    return frame;
}

//////////////////////////////////////////////
SSDMobileV2::SSDMobileV2(const std::string &path, const std::string &config, RUN_ON device)
    :BaseModel()
{
    net_ = cv::dnn::readNetFromTensorflow(path, config);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
}

cv::Mat SSDMobileV2::process(const cv::Mat &frame) {
    net_.setInput(cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(), true, false));

    cv::Mat ret = net_.forward();
    cv::Mat slice(ret.size[2], ret.size[3], CV_32F, ret.ptr<uchar>(0, 0));
    for (int i = 0; i < slice.rows; ++i) {
        const float * row = slice.ptr<float>(i);
        auto score = row[2];
        if (score > 0.6) {
            auto x = static_cast<int>(row[3] * frame.cols);
            auto y = static_cast<int>(row[4] * frame.rows);
            auto x2 = static_cast<int>(row[5] * frame.cols);
            auto y2 = static_cast<int>(row[6] * frame.cols);
            cv::rectangle(frame, cv::Rect(x, y, x2 - x, y2 - y), cv::Scalar(255, 0, 0));
        }
    }
    return frame;
}

//////////////////////////////////////////////
FasterRCNNInceptionV2::FasterRCNNInceptionV2(const std::string &path, const std::string &config, RUN_ON device)
    :BaseModel()
{
    net_ = cv::dnn::readNetFromTensorflow(path, config);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
}

cv::Mat FasterRCNNInceptionV2::process(const cv::Mat &frame) {
    net_.setInput(cv::dnn::blobFromImage(frame, 1.0, cv::Size(600, 600), cv::Scalar(), true, false));

    cv::Mat ret = net_.forward();
    cv::Mat slice(ret.size[2], ret.size[3], CV_32F, ret.ptr<uchar>(0, 0));
    for (int i = 0; i < slice.rows; ++i) {
        const float * row = slice.ptr<float>(i);
        auto score = row[2];
        if (score > 0.6) {
            auto x = static_cast<int>(row[3] * frame.cols);
            auto y = static_cast<int>(row[4] * frame.rows);
            auto x2 = static_cast<int>(row[5] * frame.cols);
            auto y2 = static_cast<int>(row[6] * frame.cols);
            cv::rectangle(frame, cv::Rect(x, y, x2 - x, y2 - y), cv::Scalar(255, 0, 0));
        }
    }
    return frame;
}

//////////////////////////////////////////////
YoloV3::YoloV3(const std::string &path, const std::string &config, const std::string &/*coco_file*/, RUN_ON device) {
    // auto classes = read_classes(coco_file);
    net_ = cv::dnn::readNet(path, config);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
    output_layers_ = net_.getUnconnectedOutLayersNames();
}

cv::Mat YoloV3::process(const cv::Mat &frame) {
    constexpr const int LARGE_YOLO3_SIZES = 320;
    //IMAGE SHOULD BE NORMALIZED
    constexpr const double NORM_FACTOR = 1.0 / 255.0;
    net_.setInput(cv::dnn::blobFromImage(frame, NORM_FACTOR, cv::Size(LARGE_YOLO3_SIZES, LARGE_YOLO3_SIZES), cv::Scalar(0,0,0), true, false));

    std::vector<std::vector<cv::Mat> > ret;
    net_.forward(ret, output_layers_);

    auto width = frame.size().width;
    auto height = frame.size().height;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;

    for (const auto &l1 : ret) {
        for (const auto &l2 : l1) {
            for (int i = 0; i < l2.rows; ++i) {
                const float * row = l2.ptr<float>(i);
                //argmax() gives idx but here we need the element itself and it's index
                // Index = class
                auto value = std::max_element(&row[5], &row[l2.cols]);
                std::size_t class_id = std::distance(&row[5], value);
                if (*value > 0.3) {
                    int center_x = static_cast<int>(row[0] * width);
                    int center_y = static_cast<int>(row[1] * height);
                    int w = static_cast<int>(row[2] * width);
                    int h = static_cast<int>(row[3] * height);
                    int x = static_cast<int>(center_x - w / 2);
                    int y = static_cast<int>(center_y - h / 2);
                    bboxes.push_back(cv::Rect(x, y, w, h));
                    scores.push_back(*value);
                }
            }
        }
    }

    std::vector<int> idxs;
    cv::dnn::NMSBoxes(bboxes, scores, 0.5f, 0.4f, idxs);
    for (const auto &idx : idxs) {
        const auto &rect = bboxes[idx];
        cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
    }

    return frame;
}

std::vector<std::string> YoloV3::read_classes(const std::string &classes) const {
    std::vector<std::string> ret;
    std::ifstream ifs(classes.c_str());
    if (ifs) {
        std::string line;
        while (std::getline(ifs, line)) {
            ret.push_back(line);
        }
    }
    return ret;
}