#include "builder.h"
#include "deep_sort.h"
#include "deep_sort_tracker/tracker.h"
#include "sort_tracker/trackers_pool.h"
#include "inside_time_tracker/in_area_tracker.h"
#include "inside_time_tracker/in_area_track.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

detector::MODEL_TYPE getByExtention(const std::string &model_file) {
    auto fn = std::filesystem::path(model_file).filename().string();
    if (fn.find("yolov3-spp") == 0) {
        return detector::YoloV3SPP;
    } else if (fn.find("yolov5") == 0) {
        return detector::YoloV5;
    }
    throw std::runtime_error("Wron model");
}

void sort_tracking(const std::string &model_path, const std::string &file_name) {
    try {
        cv::VideoCapture video_stream(file_name);
        auto detector = detector::build(getByExtention(model_path), model_path, {0});
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

std::vector<cv::Point> parse_file(const std::string &file) {
    std::vector<cv::Point> points;
    std::ifstream ifs(file.c_str());
    if (ifs) {
        std::string coords;
        while (std::getline(ifs, coords)) {
            auto pos = coords.find(',');
            if (pos != std::string::npos) {
                auto x = coords.substr(0, pos);
                auto y = coords.substr(pos + 1);
                auto i_x = std::stoi(x);
                auto i_y = std::stoi(y);
                points.push_back(cv::Point(i_x, i_y));
            }
        }
    } else {
        throw std::runtime_error("can't open polygon file");
    }
    return points;
}

void deep_sort_tracking_time(const std::string &model_path, const std::string &file_name, const std::string &deep_sort_model, const std::string &points_file) {
    try {
        auto polygon = parse_file(points_file);

        cv::VideoCapture video_stream(file_name);
        std::vector<int> persons{ 0 };
        auto detector = detector::build(getByExtention(model_path), model_path, persons);
        auto deep_sort = std::make_unique<deep_sort_tracker::DeepSort>(deep_sort_model);

        constexpr const float max_cosine_distance = 0.2f;
        constexpr const int max_badget = 100;

        //INFO: for demo
        //std::vector<cv::Point> polygon{
        //    cv::Point(2805, 775),
        //    cv::Point(3380, 465),
        //    cv::Point(3830, 950),
        //    cv::Point(3160, 1680),
        //    cv::Point(2830, 1210),
        //};

        //INFO: for testing
        //std::vector<cv::Point> polygon{
        //    cv::Point(400, 475),
        //    cv::Point(1030, 475),
        //    cv::Point(1030, 880),
        //    cv::Point(400, 880)
        //};

        //INFO: just for visual checking
        std::vector<std::vector<cv::Point>> contours{ polygon };

        auto tracker = inside_area_tracker::InAreaTracker(polygon, max_cosine_distance, max_badget);
        cv::Mat frame;
        while (video_stream.read(frame)) {
            auto start = std::chrono::system_clock::now();
            auto detections = detector->inference(frame, 0.3f, 0.5f);

            auto features = deep_sort->getFeatures(frame, detections);
            tracker.predict();
            auto end = std::chrono::system_clock::now();
            auto milseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            tracker.update(features, milseconds);

            //INFO: just for visual checking
            cv::drawContours(frame, contours, 0, cv::Scalar(0, 24, 191), 2);

            for (const auto &track : tracker.getTracks()) {
                if (!track->is_confirmed() || track->time_since_update > 1) {
                    continue;
                }

                auto bbox = track->to_tlwh();
                cv::Rect rect(
                    static_cast<int>(bbox(0)),
                    static_cast<int>(bbox(1)),
                    static_cast<int>(bbox(2)),
                    static_cast<int>(bbox(3))
                );
                std::string time = "None";
                cv::Scalar color(0, 0, 255);
                if (track->getType() == deep_sort::Track::IN_AREA_TRACKER) {
                    auto time_track = std::static_pointer_cast<inside_area_tracker::InAreaTimeTrack>(track);
                    if (time_track->isInside()) {
                        color = cv::Scalar(0, 255, 0);
                        time = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(time_track->duration()).count());
                    }
                }

                cv::rectangle(frame, rect, color, 2);
                std::string str_id = std::to_string(track->track_id) + ", class ID:" + std::to_string(track->class_id) + "Time: " + time;
                cv::putText(frame, str_id, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
            }
            //auto s = frame.size();
            //cv::Mat scaled;
            //cv::resize(frame, scaled, cv::Size(static_cast<int>(s.width / 2), static_cast<int>(s.height / 2)));
            //cv::imshow("result", scaled);
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


void deep_sort_tracking(const std::string &model_path, const std::string &file_name, const std::string &deep_sort_model) {
    // "d:\viktor_project\person_detection\tensorrt_demos\build\Debug\yolov3-spp.engine" "d:\viktor_project\test_data\videos\People - 6387.mp4" "d:\viktor_project\person_detection\tensorrt_demos\build\Release\deep_sort_32.engine"
    try {
        cv::VideoCapture video_stream(file_name);
        //cv::VideoCapture video_stream("rtsp://admin:Videoanalisi1@5.158.71.164:3010/Streaming/Channels/101");
        std::vector<int> persons{ 0 };
        auto detector = detector::build(getByExtention(model_path), model_path, persons);
        auto deep_sort = std::make_unique<deep_sort_tracker::DeepSort>(deep_sort_model);

        constexpr const float max_cosine_distance = 0.2f;
        constexpr const int max_badget = 100;
        auto tracker = deep_sort::Tracker(max_cosine_distance, max_badget);
        cv::Mat frame;
        while (video_stream.read(frame)) {
            //auto start = std::chrono::system_clock::now();
            auto detections = detector->inference(frame, 0.3f, 0.5f);

            auto features = deep_sort->getFeatures(frame, detections);
            tracker.predict();
            tracker.update(features);

            //auto end = std::chrono::system_clock::now();
            //auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            //std::cout << "Frame processing time: " << int_ms.count() << " ms" << std::endl;

            for (const auto &track : tracker.getTracks()) {
                if (!track->is_confirmed() || track->time_since_update > 1) {
                    continue;
                }
                auto bbox = track->to_tlwh();
                cv::Rect rect(
                    static_cast<int>(bbox(0)),
                    static_cast<int>(bbox(1)),
                    static_cast<int>(bbox(2)),
                    static_cast<int>(bbox(3))
                );
                cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
                std::string str_id = std::to_string(track->track_id) + ", class ID:" + std::to_string(track->class_id);
                cv::putText(frame, str_id, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            }
            auto s = frame.size();
            cv::Mat scaled;
            cv::resize(frame, scaled, cv::Size(static_cast<int>(s.width / 2), static_cast<int>(s.height / 2)));
            cv::imshow("result", scaled);
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
        } else if (type == "-t") {
            std::string model_path(argv[2]);
            std::string file_path(argv[3]);
            std::string deep_sort_model(argv[4]);
            std::string polygon_file(argv[5]);
            deep_sort_tracking_time(model_path, file_path, deep_sort_model, polygon_file);
        } else {
            std::cout << "Usage : [-d, -s, -t] \"detection_model_path\" \"video_file_path\" [\"deep_sort_model_path\"]" << std::endl;
            std::cout << "First params determines use SORT(-s option) algo or DEEP SORT(-d option)" << std::endl;
            std::cout << "\tExample: simple_demo -d \"yolov3.engine\" \"video.mp4\" \"deep_sort_32.engine\"" << std::endl;
            std::cout << "\tExample: simple_demo -s \"yolov3.engine\" \"video.mp4\"" << std::endl;
            std::cout << "\tExample: simple_demo -t \"yolov3.engine\" \"video.mp4 or stream\" \"deep_sort_32.engine\" \"file_polygon.txt\"" << std::endl;
        }
    } else {
        std::cout << "Usage : [-d, -s] \"detection_model_path\" \"video_file_path\" [\"deep_sort_model_path\"]" << std::endl;
        std::cout << "First params determines use SORT(-s option) algo or DEEP SORT(-d option)" << std::endl;
        std::cout << "\tExample: simple_demo -d \"yolov3.engine\" \"video.mp4\" \"deep_sort_32.engine\"" << std::endl;
        std::cout << "\tExample: simple_demo -s \"yolov3.engine\" \"video.mp4\"" << std::endl;
        std::cout << "\tExample: simple_demo -t \"yolov3.engine\" \"video.mp4 or stream\" \"deep_sort_32.engine\" \"file_polygon.txt\"" << std::endl;
    }
    return 0;
}