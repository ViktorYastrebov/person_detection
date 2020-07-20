#include <iostream>
#include <memory>

#include "argparser/argparser.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include "yolov3_model.h"
#include "yolov4_model.h"
#include "yolo_tiny.h"
#include "yolov3_spp.h"

//#include "tracker/deep_sort.h"

#include "trackers_pool.h"
#include "utils.h"
#include <filesystem>
#include <fstream>

std::map<std::string, int> readClasses(const std::filesystem::path &coco_classes_path) {
    std::ifstream ifs(coco_classes_path.string());
    if (ifs) {
        std::map<std::string, int> classes;
        std::string value;
        int idx = 0;
        while (std::getline(ifs, value)) {
            classes[value] = idx;
            ++idx;
        }
        return classes;
    }
    return {};
}


std::unique_ptr<BaseModel> builder(const std::string &name, const std::string &base_dir, const std::string &confidence, const std::vector<int> &classes, RUN_ON on) {
    // YoloV3, YoloV4
    float conf = 0.3f;
    try {
        conf = std::stof(confidence);
    }
    catch (const std::exception&)
    {
    }

    if (name == "YoloV3") {
        std::string w = base_dir + "/yolo_v3/yolov3.weights";
        std::string c = base_dir + "/yolo_v3/yolov3.cfg";
        return std::make_unique<YoloV3>(w, c, classes, conf, on);
    } else if (name == "YoloV4") {
        std::string w = base_dir + "/yolo_v4/yolov4.weights";
        std::string c = base_dir + "/yolo_v4/yolov4.cfg";
        return std::make_unique<YoloV4>(w, c, classes, conf, on);
    } else if (name == "YoloTiny") {
        std::string w = base_dir + "/yolov3_tiny/yolov3-tiny.weights";
        std::string c = base_dir + "/yolov3_tiny/yolov3-tiny.cfg";
        return std::make_unique<YoloV3Tiny>(w, c, classes, conf, on);
    } else if (name == "YoloV3SPP") {
        std::string w = base_dir + "/yolo3-spp/yolov3-spp.onnx";
        return std::make_unique<YoloV3SPP>(w, classes, conf, on);
    }
    return nullptr;
}

std::unique_ptr<BaseModel> builder(const std::string &name, const std::filesystem::path &base_dir, float conf, const std::vector<int> &classes, RUN_ON on) {
    if (name == "YoloV3") {
        auto w = base_dir / "yolo_v3/yolov3.weights";
        auto c = base_dir / "yolo_v3/yolov3.cfg";
        return std::make_unique<YoloV3>(w.string(), c.string(), classes, conf, on);
    } else if (name == "YoloV4") {
        auto w = base_dir / "yolo_v4/yolov4.weights";
        auto c = base_dir / "yolo_v4/yolov4.cfg";
        return std::make_unique<YoloV4>(w.string(), c.string(), classes, conf, on);
    } else if (name == "YoloTiny") {
        auto w = base_dir / "yolov3_tiny/yolov3-tiny.weights";
        auto c = base_dir / "yolov3_tiny/yolov3-tiny.cfg";
        return std::make_unique<YoloV3Tiny>(w.string(), c.string(), classes, conf, on);
    } else if (name == "YoloV3SPP") {
        auto w = base_dir / "yolo3-spp/yolov3-spp.onnx";
        return std::make_unique<YoloV3SPP>(w.string(), classes, conf, on);
    }
    return nullptr;
}

// INFO: for debug only
void process_single_image(const std::string &file, const std::string &name) {
    auto MODELS_BASE_DIR = std::filesystem::current_path() / "models";
    setBestCUDADevice();
    auto model = builder(name, MODELS_BASE_DIR.string(), 0.3f, {0}, RUN_ON::GPU);

    // auto deep_sort_path = MODELS_BASE_DIR / "deep_sort_mars_128" / "deep_sort_10.onnx";
    // auto deep_sort_path = MODELS_BASE_DIR / "deep_sort_mars_128" / "deep_sort_modified.onnx";
    auto deep_sort_path = MODELS_BASE_DIR / "pytorch_deep_sort" / "deep_sort.onnx";
    
    //DeepSortModel deepSort(deep_sort_path.string(), RUN_ON::GPU);

    if (model) {
        cv::Mat frame = cv::imread(file);
        auto output = model->process(frame);

        //auto features = deepSort.getFeatures(frame, output);

        //for (const auto &bbox : output) {
        //    cv::rectangle(frame, bbox.bbox, cv::Scalar(255, 0, 0), 2);
        //}
        //cv::imwrite("out.png", frame);
    }
}

void process_video_stream(const std::string &file, const std::string &name, const std::string &confidence) {
    auto BASE_DIR = std::filesystem::current_path() / "models";

    //////////////////////////////////////
    //INFO: HERE YOU MAY ADD ANY UI USER INTERACTION FOR CLASS SELECTING 
    //      For now it's just the demo which shows the possibilities
    //////////////////////////////////////
    auto path = BASE_DIR / "coco.names";
    auto class_map = readClasses(path);
    std::map<int, std::string> id_classes;
    std::map<int, cv::Scalar> classIdToColor;

    for (const auto &elem : class_map) {
        id_classes[elem.second] = elem.first;
        classIdToColor[elem.second] = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    }

    std::map<int, std::string>::const_iterator it = id_classes.cbegin();
    for (it; it != id_classes.cend();) {
        std::cout << std::setw(30) << it->second << " = " << it->first;
        ++it;
        if (it != id_classes.cend()) {
            std::cout << std::setw(30) << it->second << " = " << it->first << std::endl;
            ++it;
        }
    }
    std::cout << "Enter class IDs like 1,2,3,4,5 etc" << std::endl;
    std::string class_ids;
    std::getline(std::cin, class_ids);

    std::vector<int> classes;
    std::stringstream ss(class_ids);
    while (ss.good()) {
        std::string idStr;
        try {
            std::getline(ss, idStr, ',');
            int id = std::stoi(idStr);
            classes.push_back(id);
        }
        catch (const std::exception &) {
            std::cout << "Invalid Id has occured" << std::endl;
            return;
        }
    }

    //////////////////////////////////////
    setBestCUDADevice();
    auto model = builder(name, BASE_DIR.string(), confidence, classes, RUN_ON::GPU);

    if (model) {
        cv::Mat frame;
        cv::VideoCapture stream(file);

        tracker::TrackersPool tracks(10);

        cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
        {
            while (stream.read(frame)) {
                auto start = std::chrono::system_clock::now();
                auto output = model->process(frame);
                auto end = std::chrono::system_clock::now();
                auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "Detection time: " << int_ms.count() << " ms" << std::endl;

                for (const auto &bbox : output) {
                    cv::rectangle(frame, bbox.bbox, cv::Scalar(255, 0, 0), 2);
                }

                ///// THIS IS ONLY THE VISUALIZATION METHOD
                ///// YOU CAN MAKE YOUR OWN !!!
                if (output.size() > 0) {
                    auto overlay = frame.clone();
                    auto tracks_output = tracks.update(output);
                    for (const auto &result : tracks_output) {
                        auto class_color = classIdToColor[result.class_id];
                        cv::rectangle(frame, result.bbox, class_color, 1);
                        std::string str = std::to_string(result.id) + " " + id_classes[result.class_id];

                        constexpr int TILE_HEIGHT = 10;
                        cv::Rect tileRect;
                        if ((result.bbox.y - TILE_HEIGHT) > 0) {
                            tileRect.x = result.bbox.x;
                            tileRect.y = result.bbox.y - TILE_HEIGHT;
                            tileRect.width = result.bbox.width;
                            tileRect.height = TILE_HEIGHT;
                        } else {
                            tileRect.y = result.bbox.y + result.bbox.height;
                            tileRect.x = result.bbox.x;
                            tileRect.width = result.bbox.width;
                            tileRect.height = TILE_HEIGHT;
                        }
                        //INFO: It's not the best way to draw the labels, under OpenCV interface
                        //      it's hard to fit the text to the box, so I leave the constants for now
                        cv::rectangle(overlay, tileRect, class_color, cv::FILLED);
                        cv::Point p(tileRect.x, tileRect.y + 5);
                        cv::putText(frame, str, p, 0, 0.3, (127, 127, 127), 2); //VERY COST OP
                    }
                    auto alpha = 0.4;
                    cv::addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame);
                }

                //////////////////////////////////
                auto total_end = std::chrono::system_clock::now();
                auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - start);
                std::cout << "Total time: " << total_ms.count() << " ms" << std::endl;
                cv::imshow("result", frame);
                int key = cv::waitKey(1);
                if (key == 27) {
                    break;
                }

            }
        }
        cv::destroyAllWindows();
    }
}

int main(int argc, char *argv[]) {
    ap::parser p(argc, argv);
    p.add("-n", "--name", "Model name: [YoloV3, YoloV4]", ap::mode::REQUIRED);
    //p.add("-f", "--file", "Path to video file", ap::mode::REQUIRED);
    //p.add("-c", "--confidience", "confidence threshold for detection model (range [0.0, 1.0], default = 0.3)", ap::mode::OPTIONAL);

    auto args = p.parse();
    if (!args.parsed_successfully()) {
        std::cout << "Usage demo_cpp.exe -n[\"YoloV3\" | \"YoloV4\" | \"YoloTiny\"] -f \"path_to_file\" -c x.x" << std::endl;
        std::cout << std::setw(12) << " where -c is confidience threshold, range [0.0, 1.0]" << std::endl;
        return 0;
    }

    std::string test_img = "d:/viktor_project/person_detection/test_img.png";
    //process_video_stream(args["-f"], args["-n"], args["-c"]);
    process_single_image(test_img, args["-n"]);

    return 0;
}
