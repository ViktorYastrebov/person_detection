#include "builder.h"
#include "deep_sort_tracker/tracker.h"
#include "deep_sort_tracker/time_tracker.h"
#include "deep_sort_tracker/time_track.h"
#include "deep_sort_tracker/deep_sort.h"
#include "sort_tracker/tracker.h"
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

std::map<std::string, int> readClasses(const std::filesystem::path &coco_classes_path) {
    std::ifstream ifs(coco_classes_path.string());
    if (ifs) {
        std::map<std::string, int> classes;
        std::string value;
        int idx = 0;
        while (std::getline(ifs, value)) {
            classes[value] = idx;
            ++idx;
        }
        return classes;
    }
    return {};
}


std::vector<int> selectDetectionClasses(const std::string &coco_file, std::map<int, std::string> &id_classes) {
    auto class_map = readClasses(coco_file);
    //std::map<int, std::string> id_classes;
    std::map<int, cv::Scalar> classIdToColor;

    for (const auto &elem : class_map) {
        id_classes[elem.second] = elem.first;
        classIdToColor[elem.second] = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    }

    std::map<int, std::string>::const_iterator it = id_classes.cbegin();
    for (it; it != id_classes.cend();) {
        std::cout << std::setw(30) << it->second << " = " << it->first;
        ++it;
        if (it != id_classes.cend()) {
            std::cout << std::setw(30) << it->second << " = " << it->first << std::endl;
            ++it;
        }
    }
    std::cout << "Enter class IDs like 1,2,3,4,5 etc" << std::endl;
    std::string class_ids;
    std::getline(std::cin, class_ids);

    std::vector<int> classes;
    std::stringstream ss(class_ids);
    while (ss.good()) {
        std::string idStr;
        try {
            std::getline(ss, idStr, ',');
            int id = std::stoi(idStr);
            classes.push_back(id);
        }
        catch (const std::exception &) {
            std::cout << "Invalid Id has occured" << std::endl;
            return {};
        }
    }
    return classes;
}


void sort_tracking(const std::string &model_path, const std::string &file_name, const std::string &coco_classes_file, const std::string &out_name = "") {
    try {
        std::map<int, std::string> id_classes;
        cv::VideoCapture video_stream(file_name);
        auto classes = selectDetectionClasses(coco_classes_file, id_classes);
        auto detector = detector::build(getByExtention(model_path), model_path, classes);
        auto tracker = sort_tracker::Tracker(60, 3, 30);

        cv::VideoWriter writter;
        cv::Size out_size = cv::Size((int)video_stream.get(cv::CAP_PROP_FRAME_WIDTH),
            (int)video_stream.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (!out_name.empty()) {
            writter.open(out_name, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, out_size, true);
        }

        constexpr const int VECTOR_MULTIPLIER = 10;

        cv::Mat frame;
        while (video_stream.read(frame)) {
            auto start = std::chrono::system_clock::now();
            auto detections = detector->inference(frame, 0.3f, 0.5f);

            auto rets = tracker.update(detections);
            for (const auto &track : rets) {
                cv::rectangle(frame, track.bbox, cv::Scalar(0, 0, 255), 2);
                std::string str_id = std::to_string(track.id) + ", class :" + id_classes.at(track.class_id);
                cv::putText(frame, str_id, cv::Point(track.bbox.x, track.bbox.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                cv::Point2i p1(track.bbox.x + track.bbox.width / 2, track.bbox.y + track.bbox.height / 2);
                cv::Point2i p2(p1.x + track.vx * VECTOR_MULTIPLIER, p1.y + track.vy * VECTOR_MULTIPLIER);
                cv::arrowedLine(frame, p1, p2, cv::Scalar(255, 0, 0), 2);
            }
            cv::imshow("result", frame);
            int key = cv::waitKey(1);
            if (key == 27) {
                break;
            }
            writter.write(frame);
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

void deep_sort_tracking_time(const std::string &model_path, const std::string &file_name, const std::string &deep_sort_model,
                            const std::string &points_file, const std::string &coco_classes_file, const std::string &out_name) {
    try {
        std::map<int, std::string> id_classes;
        auto polygon = parse_file(points_file);
        auto classes = selectDetectionClasses(coco_classes_file, id_classes);
        cv::VideoCapture video_stream(file_name);
        auto detector = detector::build(getByExtention(model_path), model_path, classes);
        auto deep_sort = std::make_unique<deep_sort::DeepSort>(deep_sort_model);

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

        cv::VideoWriter writter;
        cv::Size out_size = cv::Size((int)video_stream.get(cv::CAP_PROP_FRAME_WIDTH),
            (int)video_stream.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (!out_name.empty()) {
            writter.open(out_name, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, out_size, true);
        }

        //INFO: just for visual checking
        std::vector<std::vector<cv::Point>> contours{ polygon };

        auto tracker = deep_sort::TimeTracker(polygon, max_cosine_distance, max_badget);
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

                auto rets = track->to_tlwh();
                cv::Rect rect(
                    static_cast<int>(rets.position(0)),
                    static_cast<int>(rets.position(1)),
                    static_cast<int>(rets.position(2)),
                    static_cast<int>(rets.position(3))
                );

                std::string time = "None";
                cv::Scalar color(0, 0, 255);
                if (track->getType() == deep_sort::Track::TIME_TRACKER) {
                    auto time_track = std::static_pointer_cast<deep_sort::TimeTrack>(track);
                    if (time_track->isInside()) {
                        color = cv::Scalar(0, 255, 0);
                        time = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(time_track->duration()).count());
                    }
                }

                cv::rectangle(frame, rect, color, 2);
                std::string str_id = std::to_string(track->track_id) + ", class :" + id_classes.at(track->class_id) + "Time: " + time;
                cv::putText(frame, str_id, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
            }
            cv::imshow("result", frame);
            int key = cv::waitKey(1);
            if (key == 27) {
                break;
            }
            writter.write(frame);
        }
    }
    catch (const std::exception &ex) {
        std::cout << "Error occured: " << ex.what() << std::endl;
    }
    catch (...) {
        std::cout << "Unhandled exception" << std::endl;
    }
}


void deep_sort_tracking(const std::string &model_path, const std::string &file_name, const std::string &deep_sort_model, const std::string &coco_classes_file, const std::string &out_name) {
    // "d:\viktor_project\person_detection\tensorrt_demos\build\Debug\yolov3-spp.engine" "d:\viktor_project\test_data\videos\People - 6387.mp4" "d:\viktor_project\person_detection\tensorrt_demos\build\Release\deep_sort_32.engine"
    try {
        std::map<int, std::string> id_classes;
        auto classes = selectDetectionClasses(coco_classes_file, id_classes);
        cv::VideoCapture video_stream(file_name);
        auto detector = detector::build(getByExtention(model_path), model_path, classes);
        auto deep_sort = std::make_unique<deep_sort::DeepSort>(std::filesystem::path(deep_sort_model));

        constexpr const int VECTOR_MULTIPLIER = 20;
        constexpr const float max_cosine_distance = 0.2f;
        constexpr const int max_badget = common::datatypes::FEATURES_SIZE;

        auto tracker = deep_sort::Tracker(max_cosine_distance, max_badget, 0.7f, 120, 3, 3, 60);
        cv::Mat frame;

        cv::VideoWriter writter;
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        int ex = static_cast<int>(video_stream.get(cv::CAP_PROP_FOURCC));
        cv::Size out_size = cv::Size((int)video_stream.get(cv::CAP_PROP_FRAME_WIDTH),
                              (int)video_stream.get(cv::CAP_PROP_FRAME_HEIGHT));

        std::cout << "output size : " << out_size.width << ", " << out_size.height << std::endl;

        if (!out_name.empty()) {
            writter.open(out_name, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, out_size, true);
        }

        while (video_stream.read(frame)) {
            //auto start = std::chrono::system_clock::now();
            // orig: 0.3, 0.5f -> 0.5 0.7
            auto detections = detector->inference(frame, 0.3f, 0.5f);

            auto features = deep_sort->getFeatures(frame, detections);
            tracker.predict();
            tracker.update(features);

            //auto end = std::chrono::system_clock::now();
            //auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            //std::cout << "Frame processing time: " << int_ms.count() << " ms" << std::endl;

            for (const auto &track : tracker.getTracks()) {
#if 0
                if (!track->is_confirmed() || track->time_since_update > 1) {
                    continue;
                }
#endif

#if 1
                if (!track->is_confirmed()) {
                    continue;
                }
#endif

                auto rets = track->to_tlwh();
                cv::Rect rect(
                    static_cast<int>(rets.position(0)),
                    static_cast<int>(rets.position(1)),
                    static_cast<int>(rets.position(2)),
                    static_cast<int>(rets.position(3))
                );

                cv::Point rectCenter(rect.x + rect.width / 2, rect.y + rect.height / 2);
                cv::Point velCenter(static_cast<int>(rets.velocity(0) + rets.velocity(2) / 2.0f),
                                    static_cast<int>(rets.velocity(1) + rets.velocity(3) / 2.0f)
                );

                cv::Point p1(rectCenter.x, rectCenter.y);
                cv::Point p2(rectCenter.x + velCenter.x * VECTOR_MULTIPLIER, rectCenter.y + velCenter.y * VECTOR_MULTIPLIER);
                cv::arrowedLine(frame, p1, p2, cv::Scalar(255, 0, 0), 2);

                cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
                std::string str_id = std::to_string(track->track_id) + ", class :" + id_classes.at(track->class_id);
                cv::putText(frame, str_id, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            }
            cv::imshow("result", frame);
            int key = cv::waitKey(1);
            if (key == 27) {
                break;
            }
            writter.write(frame);
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
            std::string coco_classes_file(argv[5]);
            std::string out_file;
            if (argc > 6) {
                out_file = std::string(argv[6]);
                std::cout << "Out file: " << out_file << std::endl;
            }
            deep_sort_tracking(model_path, file_path, deep_sort_model, coco_classes_file, out_file);
        } else if (type == "-s") {
            std::string model_path(argv[2]);
            std::string file_path(argv[3]);
            std::string coco_classes_file(argv[4]);
            std::string out_file;
            if (argc > 5) {
                out_file = std::string(argv[5]);
            }
            sort_tracking(model_path, file_path, coco_classes_file, out_file);
        } else if (type == "-t") {
            std::string model_path(argv[2]);
            std::string file_path(argv[3]);
            std::string deep_sort_model(argv[4]);
            std::string polygon_file(argv[5]);
            std::string coco_classes_file(argv[6]);
            std::string out_file;
            if (argc > 7) {
                out_file = std::string(argv[7]);
            }
            deep_sort_tracking_time(model_path, file_path, deep_sort_model, polygon_file, coco_classes_file, out_file);
        } else {
            std::cout << "Usage : [-d, -s, -t] \"detection_model_path\" \"video_file_path\" [\"deep_sort_model_path\"]" << std::endl;
            std::cout << "First params determines use SORT(-s option) algo or DEEP SORT(-d option) or DEEP SORT TIME(-t option)" << std::endl;
            std::cout << "\tExample: simple_demo -d \"yolov3.engine\" \"video.mp4\" \"deep_sort_32.engine\"" << std::endl;
            std::cout << "\tExample: simple_demo -s \"yolov3.engine\" \"video.mp4\"" << std::endl;
            std::cout << "\tExample: simple_demo -t \"yolov3.engine\" \"video.mp4 or stream\" \"deep_sort_32.engine\" \"file_polygon.txt\"" << std::endl;
        }
    } else {
        std::cout << "Usage : [-d, -s] \"detection_model_path\" \"video_file_path\" [\"deep_sort_model_path\"]" << std::endl;
        std::cout << "First params determines use SORT(-s option) algo or DEEP SORT(-d option) or DEEP SORT TIME(-t option)" << std::endl;
        std::cout << "\tExample: simple_demo -d \"yolov3.engine\" \"video.mp4\" \"deep_sort_32.engine\"" << std::endl;
        std::cout << "\tExample: simple_demo -s \"yolov3.engine\" \"video.mp4\"" << std::endl;
        std::cout << "\tExample: simple_demo -t \"yolov3.engine\" \"video.mp4 or stream\" \"deep_sort_32.engine\" \"file_polygon.txt\"" << std::endl;
    }
    return 0;
}