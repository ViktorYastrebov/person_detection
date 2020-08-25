#include "yolov3_model.h"
#include "yolov5_model.h"
#include "deep_sort.h"
#include "deep_sort_tracker/tracker.h"
#include "sort_tracker/trackers_pool.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>


void sort_tracking(int argc, char*argv[]) {
    try {
        std::string model_path(argv[1]);
        std::string file_name(argv[2]);

        cv::VideoCapture video_stream(file_name);
        auto detector = std::make_unique<detector::YoloV3SPPModel>(model_path);
        auto tracker = sort_tracker::TrackersPool(10);

        cv::Mat frame;
        while (video_stream.read(frame)) {
            auto start = std::chrono::system_clock::now();
            auto detections = detector->inference(frame, 0.3f, 0.5f);

            auto rets = tracker.update(detections);
            //tracker.update(detections);
            for (const auto &track : rets) {
                //cv::Rect cv_rect(bbox(0), bbox(1), bbox(2), bbox(3));
                cv::rectangle(frame, track.bbox, cv::Scalar(0, 0, 255), 2);
                std::string str_id = std::to_string(track.id);
                cv::putText(frame, str_id, cv::Point(track.bbox.x, track.bbox.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            }
            cv::imshow("result", frame);
            int key = cv::waitKey(1);
            if (key == 27) {
                break;
            }
        }
    } catch (const std::exception &ex) {
        std::cout << "Error occured: " << ex.what() << std::endl;
    } catch (...) {
        std::cout << "Unhandled exception" << std::endl;
    }
}


void deep_sort_tracking(int argc, char*argv[]) {
    // "d:\viktor_project\person_detection\tensorrt_demos\build\Debug\yolov3-spp.engine" "d:\viktor_project\test_data\videos\People - 6387.mp4" "d:\viktor_project\person_detection\tensorrt_demos\build\Release\deep_sort_32.engine"
    try {
        std::string model_path(argv[1]);

        std::string file_name(argv[2]);
        std::filesystem::path deep_sort_model(argv[3]);

        cv::VideoCapture video_stream(file_name);
        auto detector = std::make_unique<detector::YoloV3SPPModel>(model_path);
        auto deep_sort = std::make_unique<deep_sort_tracker::DeepSort>(deep_sort_model);

        constexpr const float max_cosine_distance = 0.2;
        constexpr const int max_badget = 100;
        auto tracker = deep_sort::Tracker(max_cosine_distance, max_badget);
        cv::Mat frame;
        while (video_stream.read(frame)) {
            auto start = std::chrono::system_clock::now();
            auto detections = detector->inference(frame, 0.3f, 0.5f);

            auto features = deep_sort->getFeatures(frame, detections);
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
    }
    catch (const std::exception &ex) {
        std::cout << "Error occured: " << ex.what() << std::endl;
    }
    catch (...) {
        std::cout << "Unhandled exception" << std::endl;
    }
}


int main(int argc, char *argv[]) {
    sort_tracking(argc, argv);
    //deep_sort_tracking(argc, argv);
    return 0;
}