#include <iostream>
#include <memory>

#include "argparser/argparser.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>

#include "model_producer.h"
#include "model.h"

void process_video_stream(const std::string &file, const std::string &model_name) {
    ModelProducer mp;
    auto model = mp.get(model_name, RUN_ON::GPU);
    if (model) {
        cv::Mat frame;
        cv::VideoCapture stream(file);

        cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
        {
            while (stream.read(frame)) {
                cv::Mat ret = model->process(frame);
                cv::imshow("result", ret);
                cv::waitKey(1);
            }
        }
        cv::destroyAllWindows();
    }
}

int main(int argc, char *argv[]) {
    ap::parser p(argc, argv);
    p.add("-f", "--file", "Path to video file", ap::mode::REQUIRED);
    p.add("-m", "--model", "Select 1 of the models: ssdlite_mobilenet_v2_coco,\nssd_mobilenet_v2_coco,\nfast_rcnn_v2_coco,\nyolov3_coco");

    auto args = p.parse();
    if (!args.parsed_successfully()) {
        return 0;
    }

    auto file = args["-f"];
    std::string model_name = "ssdlite_mobilenet_v2_coco";

    if(!args["-m"].empty()) {
        model_name = args["-m"];
    }
    process_video_stream(file, model_name);

    return 0;
}
