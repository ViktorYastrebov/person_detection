#include "model_producer.h"
#include "model.h"

#include <filesystem>
#include <vector>

const std::string models[] = {
        "ssdlite_mobilenet_v2_coco",
        "ssd_mobilenet_v2_coco",
        "fast_rcnn_v2_coco",
        "yolov3_coco"
};

std::unique_ptr< BaseModel > ModelProducer::get(const std::string &model_name, RUN_ON on) {
    if (model_name == models[0]) {
        std::string model_path = "./models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb";
        std::string config_path = "./models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pbtxt";
        return std::make_unique<SSDliteMobileV2>(model_path, config_path, on);
    } else if (model_name == models[1]) {
        std::string model_path = "./models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
        std::string config_path = "./models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pbtxt";
        return std::make_unique< SSDMobileV2 >(model_path, config_path, on);
    } else if (model_name == models[2]) {
        std::string model_path = "./models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
        std::string config_path = "./models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pbtxt";
        return std::make_unique< FasterRCNNInceptionV2>(model_path, config_path, on);
    } else if (model_name == models[3]) {
        //std::string model_path = "./models/yolo_v3/model.pb";
        std::string model_weights = "./models/yolo_v3/yolov3.weights";
        std::string model_config = "./models/yolo_v3/yolov3.cfg";
        std::string coco_classes = "./models/yolo_v3/coco.names";
        return std::make_unique< YoloV3 >(model_weights, model_config, coco_classes, on);
    }
    return nullptr;
}
