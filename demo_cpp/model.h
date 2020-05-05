#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "device_utils.h"

struct BaseModel {
    virtual ~BaseModel() = 0
    {}
    virtual cv::Mat process(const cv::Mat &frame) = 0;
};

//Image sizes : (300, 300)
struct SSDliteMobileV2 : public BaseModel {
    SSDliteMobileV2(const std::string &path, const std::string &config = "", RUN_ON device = RUN_ON::CPU);
    virtual ~SSDliteMobileV2() = default;
    cv::Mat process(const cv::Mat &frame);
private:
    cv::dnn::Net net_;
};

//Image sizes : (300, 300)
struct SSDMobileV2 : public BaseModel {
    SSDMobileV2(const std::string &path, const std::string &config = "", RUN_ON device = RUN_ON::CPU);
    virtual ~SSDMobileV2() = default;
    cv::Mat process(const cv::Mat &frame);
private:
    cv::dnn::Net net_;
};

//Image sizes : min(600, 600), max(1024, 1024), need to see config
struct FasterRCNNInceptionV2 : public BaseModel {
    FasterRCNNInceptionV2(const std::string &path, const std::string &config = "", RUN_ON device = RUN_ON::CPU);
    virtual ~FasterRCNNInceptionV2() = default;
    cv::Mat process(const cv::Mat &frame);
private:
    cv::dnn::Net net_;
};

//Image sizes : (416, 416)
struct YoloV3 : public BaseModel {
    YoloV3(const std::string &path, const std::string &config, const std::string &classes, RUN_ON device = RUN_ON::CPU);
    virtual ~YoloV3() = default;
    cv::Mat process(const cv::Mat &frame);
private:
    std::vector<std::string> read_classes(const std::string &classes) const;
private:
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
};
