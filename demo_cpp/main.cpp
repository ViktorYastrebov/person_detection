#include <iostream>
#include <memory>

#include "argparser/argparser.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include "detection_engine/yolov3_model.h"

#include <iostream>

void process_video_stream(const std::string &file, const std::string &weights, const std::string &model_config) {
    auto model = std::make_unique<YoloV3>(weights, model_config, RUN_ON::GPU);
    if (model) {
        cv::Mat frame;
        cv::VideoCapture stream(file);

        cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
        {
            while (stream.read(frame)) {
                auto bboxes = model->process(frame);
                std::cout << "Persons on the frame : " << bboxes.size() << std::endl;
                for (const auto &bbox : bboxes) {
                    cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2);
                }
                cv::imshow("result", frame);
                cv::waitKey(1);
            }
        }
        cv::destroyAllWindows();
    }
}

int main(int argc, char *argv[]) {
    ap::parser p(argc, argv);
    p.add("-f", "--file", "Path to video file", ap::mode::REQUIRED);
    p.add("-m", "--model", "Path to model weights", ap::mode::REQUIRED);
    p.add("-c", "--config", "Path to model config");

    auto args = p.parse();
    if (!args.parsed_successfully()) {
        return 0;
    }

    auto file = args["-f"];
    process_video_stream(file, args["-m"], args["-c"]);

    return 0;
}
