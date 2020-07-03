#include "model_builder.h"
#include "base_model.h"
#include "yolov3_batch.h"
#include "yolov4_batch.h"
#include <filesystem>

ModelBuilder::ModelBuilder(const std::string &name, const std::string &conf)
    :name_(name)
    , conf_(0.3)
{
    try {
        conf_ = std::stof(conf);
    }
    catch (const std::exception&)
    {
        throw std::logic_error("wrong confidence param");
    }
}

std::unique_ptr<BaseModelBatched> ModelBuilder::build(const std::vector<std::string> &files, const std::vector<int> &classes, RUN_ON on) const {
    auto BASE_DIR = std::filesystem::current_path() / "models";
    if (name_ == "YoloV3") {
        std::string w = std::filesystem::path(BASE_DIR / "yolov3_batch/yolov3.weights").string();
        std::string c = std::filesystem::path(BASE_DIR / "yolov3_batch/yolov3.cfg").string();
        return std::make_unique<YoloV3Batched>(w, c, classes, conf_, on);
    } else if (name_ == "YoloV4") {
        std::string w = std::filesystem::path(BASE_DIR / "/yolov4_batch/yolov4.weights").string();
        std::string c = std::filesystem::path(BASE_DIR / "/yolov4_batch/yolov4.cfg").string();
        return std::make_unique<YoloV4Batched>(w, c, classes, conf_, on);
    }
    throw std::runtime_error("Unknown model name");
}