#include "generic_detector.h"


int main(int argc, char *argv[]) {
    try {
        std::string onnx_path(argv[1]);

        const std::string img_path("d:/viktor_project/person_detection/tensorrt_demos/test.png");

        detection_engine::GenericDetector detector(onnx_path, 1);
        if (detector.buildEngine()) {
            if (detector.prepareBuffers()) {
                cv::Mat img = cv::imread(img_path);
                detector.inference(img);
            }
        }
    } catch (const std::exception &ex) {
        std::cout << "Error occured: " << ex.what() << std::endl;
    }
    catch (...) {
        std::cout << "Unhandled exception" << std::endl;
    }
    return 0;
}