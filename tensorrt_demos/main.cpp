#include "generic_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    try {
        std::string model_path(argv[1]);

        std::string file_name(argv[2]);
        cv::VideoCapture video_stream(file_name);
        auto detector = std::make_unique<detection_engine::GenericDetector>(model_path);
        cv::Mat frame;
        while (video_stream.read(frame)) {
            auto start = std::chrono::system_clock::now();
            auto rects = detector->inference(frame, 0.3f, 0.5f);
            auto end = std::chrono::system_clock::now();
            auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "detection time : " << int_ms.count() << " ms" << std::endl;

            for (const auto &rect : rects) {
                cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
            }
            cv::imshow("result", frame);
            int key = cv::waitKey(1);
            if (key == 27) {
                break;
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