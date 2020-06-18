#include <iostream>
#include <memory>

#include "argparser/argparser.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include "detection_engine/yolov3_model.h"
#include "detection_engine/yolov4_model.h"

#include "detection_engine/tracker/trackers_pool.h"


std::unique_ptr<BaseModel> builder(const std::string &name, const std::string &base_dir, RUN_ON on) {
    // YoloV3, YoloV4
    if (name == "YoloV3") {
        std::string w = base_dir + "yolo_v3/yolov3.weights";
        std::string c = base_dir + "yolo_v3/yolov3.cfg";
        return std::make_unique<YoloV3>(w, c, on);
    } else if (name == "YoloV4") {
        std::string w = base_dir + "yolo_v4/yolov4.weights";
        std::string c = base_dir + "yolo_v4/yolov4.cfg";
        return std::make_unique<YoloV4>(w, c, on);
    }
    return nullptr;
}


void process_video_stream(const std::string &file, const std::string &name) {
    const std::string BASE_DIR = "d:/viktor_project/person_detection/demo_cpp/models/";
    auto model = builder(name, BASE_DIR, RUN_ON::GPU);

    if (model) {
        cv::Mat frame;
        cv::VideoCapture stream(file);

        tracker::TrackersPool tracks;

        cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
        {
            while (stream.read(frame)) {
                auto start = std::chrono::system_clock::now();
                auto bboxes = model->process(frame);
                auto end = std::chrono::system_clock::now();
                auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "Detection time: " << int_ms.count() << " ms" << std::endl;

                // std::cout << "Persons on the frame : " << bboxes.size() << std::endl;
                for (const auto &bbox : bboxes) {
                    cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2);
                }
                auto tracks_bboxes = tracks.update(bboxes);
                for (const auto &result : tracks_bboxes) {
                    cv::rectangle(frame, result.bbox, cv::Scalar(0, 0, 255), 2);
                    std::string id = std::to_string(result.id);
                    cv::Point p(result.bbox.x, result.bbox.y);
                    cv::putText(frame, id, p, 0, 5e-3 * 200, (0, 255, 0), 2);
                }
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
    p.add("-n", "--name", "Model name: [YoloV3, YoloV4]");
    p.add("-f", "--file", "Path to video file", ap::mode::REQUIRED);

    auto args = p.parse();
    if (!args.parsed_successfully()) {
        return 0;
    }
    process_video_stream(args["-f"], args["-n"]);

    return 0;
}
