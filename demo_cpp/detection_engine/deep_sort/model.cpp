#include "deep_sort/model.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

namespace deep_sort {

    ImageEncoderModel::ImageEncoderModel(const std::string &base_dir, RUN_ON device) {
        const std::string path = base_dir + "/deep_sort/mars-small128.pb";
        //const std::string cfg = base_dir + "/deep_sort/mars-small128.pbtxt";
        const std::string cfg = base_dir + "/deep_sort/freeze.pbtxt";

        net_ = cv::dnn::readNetFromTensorflow(path, cfg);
        if (device == RUN_ON::GPU) {
            net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
        }
        // Layers: 
        //      images:0: input
        //      features:0 - output
        //std::cout << "Layers count: " << net_.getLayersCount() << std::endl;

        std::cout << "out layer id: " << net_.getLayerId("out") << std::endl;
        std::cout << "in  layer id: " << net_.getLayerId("in") << std::endl;
        std::cout << "images:0  layer id: " << net_.getLayerId("images:0") << std::endl;
        std::cout << "features:0  layer id: " << net_.getLayerId("features:0") << std::endl;

        auto names = net_.getUnconnectedOutLayersNames();
        for (const auto &name : names) {
            std::cout << name << std::endl;
        }
    }

    void ImageEncoderModel::encode(cv::Mat frame, const std::vector<cv::Rect> &bboxes) {
        std::vector<cv::Mat> patches;
        for (const auto &box : bboxes) {
            patches.push_back(extract_image_patch(frame, box));
        }
        cv::Mat input = cv::dnn::blobFromImages(patches, 1.0, cv::Size(), cv::Scalar(), true, false);
        net_.setInput(input);
        cv::Mat ret = net_.forward("out");
        std::cout << ret.dims << std::endl;
    }

    cv::Mat ImageEncoderModel::extract_image_patch(cv::Mat frame, const cv::Rect &bbox) {
        cv::Rect normalized = bbox;
        //INFO: confused, must be height
        //auto new_width = static_cast<int>(std::round(bbox.height * ASPECT));
        //changed_rect.x -= static_cast<int>(std::round((new_width - changed_rect.width) / 2));
        //changed_rect.width = new_width;
        //convert to top,left, right,bottom
        cv::Size frame_size = frame.size();
        if (normalized.x < 0) {
            normalized.width += std::abs(normalized.x);
            normalized.x = 0;
        }
        if (normalized.y < 0) {
            normalized.height += std::abs(normalized.y);
            normalized.y = 0;
        }

        cv::Mat patch = frame(normalized);
        cv::resize(patch, patch, cv::Size(128, 64));
        return patch;
    }
}