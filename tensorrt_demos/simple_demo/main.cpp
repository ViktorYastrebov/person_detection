#include "builder.h"
#include "deep_sort.h"
#include "deep_sort_tracker/tracker.h"
#include "sort_tracker/trackers_pool.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>


void sort_tracking(const std::string &model_path, const std::string &file_name) {
    try {
        cv::VideoCapture video_stream(file_name);
        auto detector = detector::build(detector::YoloV3SPP, model_path);
        auto tracker = sort_tracker::TrackersPool(10);

        cv::Mat frame;
        while (video_stream.read(frame)) {
            auto start = std::chrono::system_clock::now();
            auto detections = detector->inference(frame, 0.3f, 0.5f);

            auto rets = tracker.update(detections);
            for (const auto &track : rets) {
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


void deep_sort_tracking(const std::string &model_path, const std::string &file_name, const std::string &deep_sort_model) {
    // "d:\viktor_project\person_detection\tensorrt_demos\build\Debug\yolov3-spp.engine" "d:\viktor_project\test_data\videos\People - 6387.mp4" "d:\viktor_project\person_detection\tensorrt_demos\build\Release\deep_sort_32.engine"
    try {
        cv::VideoCapture video_stream(file_name);
        auto detector = detector::build(detector::YoloV3SPP, model_path);
        auto deep_sort = std::make_unique<deep_sort_tracker::DeepSort>(deep_sort_model);

        constexpr const float max_cosine_distance = 0.2f;
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
    if (argc > 2) {
        std::string type(argv[1]);
        if (type == "-d") {
            std::string model_path(argv[2]);
            std::string file_path(argv[3]);
            std::string deep_sort_model(argv[4]);
            deep_sort_tracking(model_path, file_path, deep_sort_model);
        } else if (type == "-s") {
            std::string model_path(argv[2]);
            std::string file_path(argv[3]);
            sort_tracking(model_path, file_path);
        } else {
            std::cout << "Usage : [-d, -s] \"detection_model_path\" \"video_file_path\" [\"deep_sort_model_path\"]" << std::endl;
            std::cout << "First params determines use SORT(-s option) algo or DEEP SORT(-d option)" << std::endl;
            std::cout << "\tExample: simple_demo -d \"yolov3.engine\" \"video.mp4\" \"deep_sort_32.engine\"" << std::endl;
            std::cout << "\tExample: simple_demo -s \"yolov3.engine\" \"video.mp4\"" << std::endl;
        }
    } else {
        std::cout << "Usage : [-d, -s] \"detection_model_path\" \"video_file_path\" [\"deep_sort_model_path\"]" << std::endl;
        std::cout << "First params determines use SORT(-s option) algo or DEEP SORT(-d option)" << std::endl;
        std::cout << "\tExample: simple_demo -d \"yolov3.engine\" \"video.mp4\" \"deep_sort_32.engine\"" << std::endl;
        std::cout << "\tExample: simple_demo -s \"yolov3.engine\" \"video.mp4\"" << std::endl;
    }
    return 0;
}