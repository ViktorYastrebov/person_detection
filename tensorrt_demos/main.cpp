#include "generic_detector.h"
#include "deep_sort.h"
#include "deep_sort_tracker/tracker.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

int main(int argc, char *argv[]) {
    try {
        std::string model_path(argv[1]);

        std::string file_name(argv[2]);
        std::filesystem::path deep_sort_model(argv[3]);

        cv::VideoCapture video_stream(file_name);
        auto detector = std::make_unique<detection_engine::GenericDetector>(model_path);
        auto deep_sort = std::make_unique<deep_sort_tracker::DeepSort>(deep_sort_model);

        constexpr const float max_cosine_distance = 0.2;
        constexpr const int max_badget = 100;
        auto tracker = Tracker(max_cosine_distance, max_badget);
        cv::Mat frame;
        while (video_stream.read(frame)) {
            auto start = std::chrono::system_clock::now();
            auto rects = detector->inference(frame, 0.3f, 0.5f);
            
            auto features = deep_sort->getFeatures(frame, rects);
            tracker.predict();
            tracker.update(features);

            auto end = std::chrono::system_clock::now();
            auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Frame processing time: " << int_ms.count() << " ms" << std::endl;

            for (const auto &track : tracker.getTracks()) {
                if (!track.is_confirmed() || track.time_since_update > 1) {
                    continue;
                }
                auto bbox = track.to_tlwh();
                cv::Rect rect(
                    static_cast<int>(bbox(0)),
                    static_cast<int>(bbox(1)),
                    static_cast<int>(bbox(2)),
                    static_cast<int>(bbox(3))
                );
                cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
                std::string str_id = std::to_string(track.track_id);
                cv::putText(frame, str_id, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
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