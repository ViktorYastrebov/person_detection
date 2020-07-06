#include "main_window.h"
#include "model_builder.h"
#include "base_model.h"
#include "track_processor.h"
#include <opencv2/videoio.hpp>

MainWindow::MainWindow()
    :QMainWindow()
    , central_widget_(this)
{
    setCentralWidget(&central_widget_);
}

MainWindow::~MainWindow()
{
    //INFO: might need force stop
    if (runner_.joinable()) {
        runner_.join();
    }
}

void MainWindow::Process(const std::string &name, const std::string &conf, const std::vector<std::string> &files) {
    auto func = std::bind(&MainWindow::ProcessImpl, this, name, conf, files);
    runner_ = std::thread(func);
}

void MainWindow::ProcessImpl(const std::string &name, const std::string &conf, const std::vector<std::string> &files) {
    try {
        ModelBuilder builder(name, conf);
        auto model = builder.build(files, { 0 }, RUN_ON::GPU);
        sendOutput("model loaded");

        using Stream2TrackType = std::map<std::shared_ptr<cv::VideoCapture>, std::shared_ptr<TrackProcessor>>;
        Stream2TrackType streams2tracks;

        int idx = 0;
        for (const auto &file : files) {
            auto stream = std::make_shared<cv::VideoCapture>(file);
            auto track = std::make_shared<TrackProcessor>(idx, central_widget_);
            streams2tracks.insert(std::pair(stream, track));
            track->start();
            ++idx;
        }

        bool processing = true;
        while (processing) {
            std::vector<cv::Mat> frames;
            std::vector<int> out_idxs;

            auto it = streams2tracks.begin();
            while(it != streams2tracks.end()) {
                cv::Mat frame;
                if (it->first->read(frame)) {
                    frames.push_back(frame);
                    //out_idxs.push_back(it->second);
                    ++it;
                } else {
                    it->second->stop();
                    central_widget_.stopView(it->second->getId());
                    it = streams2tracks.erase(it);
                }
            }

            if (frames.empty()) {
                processing = false;
                return;
            }

            auto start = std::chrono::system_clock::now();
            {
                //Here need to pass correct output to track
                Stream2TrackType::iterator it = streams2tracks.begin();
                auto multi_output = model->process(frames);
                for (it; it != streams2tracks.end(); ++it) {
                    it->second->put({ frames[it->second->getId()], multi_output[it->second->getId()] });
                }
#if 0
                auto multi_output = model->process(frames);
                for (int idx = 0; idx < multi_output.size(); ++idx) {
                    central_widget_.putTo(out_idxs[idx], frames[idx], multi_output[idx]);
                }
#endif
            }
            auto end = std::chrono::system_clock::now();
            auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::string msg = "Detection time: " + std::to_string(int_ms) + " ms";
            sendOutput(msg);
        }
    }
    catch (const std::string &) {
    }
}

void MainWindow::sendOutput(const std::string &message) {
    auto value_copy = std::string(message);
    central_widget_.sendOutput(message);
}
