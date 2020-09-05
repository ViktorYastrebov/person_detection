#include <opencv2/videoio.hpp>
#include <QKeyEvent>
#include "main_window.h"

//#include "detection_engine/generic_detector.h"
#include "detection_engine/yolov3_model.h"
#include "detection_engine/deep_sort.h"
#include "deep_sort_tracker/tracker.h"

#include <fstream>

MainWindow::MainWindow(const std::filesystem::path &detector_path, const std::filesystem::path &deep_sort_path)
    :QMainWindow()
    , detector_path_(detector_path)
    , deep_sort_path_(deep_sort_path)
    , display_frame_(this)
    , stop_(true)
{
    setCentralWidget(&display_frame_);
}

//INFO: might need force stop
MainWindow::~MainWindow()
{
    if (runner_.joinable()) {
        runner_.join();
    }
}

void MainWindow::Process(const std::filesystem::path &video_file) {
    stop_ = false;
    auto func = std::bind(&MainWindow::ProcessImpl, this, video_file);
    runner_ = std::thread(func);
}

void MainWindow::ProcessImpl(const std::filesystem::path &file_path) {
    std::ofstream trace("trace.txt");
    try {
        std::vector<int> person_class{ 0 };
        auto detector = std::make_unique<detector::YoloV3SPPModel>(detector_path_, person_class);
        auto feature_extractor = std::make_unique< deep_sort_tracker::DeepSort>(deep_sort_path_);
        auto video_stream = cv::VideoCapture(file_path.string());

        constexpr const float max_cosine_distance = 0.2f;
        constexpr const int max_badget = 100;
        auto tracker = deep_sort::Tracker(max_cosine_distance, max_badget);
        cv::Mat frame;

        while (video_stream.read(frame)) {
            auto detections = detector->inference(frame, 0.3f, 0.5f);

            auto features = feature_extractor->getFeatures(frame, detections);
            tracker.predict();
            tracker.update(features);

            auto tracks = tracker.getTracks();

            DisplayFrame::InputData data;
            data.frame = frame;

            for (const auto &track : tracker.getTracks()) {
                if (!track->is_confirmed() || track->time_since_update > 1) {
                    continue;
                }
                auto bbox = track->to_tlwh();
                data.tracks_data.push_back({ bbox, track->track_id, track->class_id });
            }
            display_frame_.putInput(std::move(data));

            if (stop_) {
                display_frame_.Stop();
                break;
            }
        }
    } catch (const std::exception &ex) {
        trace << "Exception occurs : " << ex.what() << std::endl;
    }
    catch (...) {
        trace << "Something terrible has happened" << std::endl;
    }
}


void MainWindow::keyPressEvent(QKeyEvent *ev) {
    int key = ev->key();
    if (key == Qt::Key_Escape) {
        stop_ = true;
    }
}

//void MainWindow::sendOutput(const std::string &message) {
//    auto value_copy = std::string(message);
//    central_widget_.sendOutput(message);
//}
