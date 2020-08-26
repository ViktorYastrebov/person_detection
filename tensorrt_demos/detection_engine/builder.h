#pragma once

#include "base_model.h"
#include <filesystem>

namespace detector {
    //INFO: for different models might need params wrapper + singleton map holder

    enum MODEL_TYPE {
        YoloV3SPP,
        //INFO: might need to split between YoloV5s/YoloV5m/YoloV5l/YoloV5x cause of input shapes, under python they are different
        YoloV5
    };

    std::unique_ptr<BaseDetector> ENGINE_DECL build(MODEL_TYPE type, const std::filesystem::path &model_path, const int BATCH_SIZE = 1);
}