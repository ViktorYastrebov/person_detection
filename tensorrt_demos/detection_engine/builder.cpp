#include "builder.h"
#include "yolov3_model.h"
#include "yolov5_model.h"

namespace detector {
    std::unique_ptr<BaseDetector> build(MODEL_TYPE type, const std::filesystem::path &model_path, const std::vector<int> &classes_ids, const int BATCH_SIZE) {
        if (type == MODEL_TYPE::YoloV3SPP) {
            return std::make_unique<YoloV3SPPModel>(model_path, classes_ids, BATCH_SIZE);
        } else if (type == MODEL_TYPE::YoloV5) {
            return std::make_unique<YoloV5Model>(model_path, classes_ids, BATCH_SIZE);
        }
        return nullptr;
    }
}