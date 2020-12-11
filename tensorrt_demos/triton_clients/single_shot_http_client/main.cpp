#include "utils.h"
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {

    const std::string url = "127.0.0.1:8000";
    const std::string model_name = "yolov5m";

    if(argc < 2) {
        std::cout << "usage : client image_file.png" << std::endl;
        return 0;
    }
    std::string file(argv[1]);

    try {
        std::vector<int> classes { 0 };
        cv::Mat img = cv::imread(file);
        if(img.empty()) {
            std::cout << "can't read img" << std::endl;
            return 0;
        }
        triton_inference::Client client(url, model_name, classes);
        auto detections = client.inference(img ,0.3f, 0.5f);
        for(const auto &detection : detections) {
            cv::Rect rect {
                static_cast<int>(detection.bbox(0)),
                static_cast<int>(detection.bbox(1)),
                static_cast<int>(detection.bbox(2)),
                static_cast<int>(detection.bbox(3))
            };
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
        }
        cv::imwrite("output.png", img);
    } catch ( const std::exception &ex) {
        std::cout << " Error occurs :" << ex.what() << std::endl;
    }
    return 0;
}
