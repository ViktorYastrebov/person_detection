#include "main_window.h"
#include "model_builder.h"
#include "base_model.h"
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

        std::map<std::shared_ptr<cv::VideoCapture>, int> streamIds;
        int idx = 0;
        for (const auto &file : files) {
            auto stream = std::make_shared<cv::VideoCapture>(file);
            streamIds.insert(std::pair(stream, idx));
            ++idx;
        }

        bool processing = true;
        while (processing) {
            std::vector<cv::Mat> frames;
            std::vector<int> out_idxs;

            auto it = streamIds.begin();
            while(it != streamIds.end()) {
                cv::Mat frame;
                if (it->first->read(frame)) {
                    frames.push_back(frame);
                    out_idxs.push_back(it->second);
                    ++it;
                } else {
                    central_widget_.stopView(it->second);
                    it = streamIds.erase(it);
                }
            }

            if (out_idxs.empty()) {
                processing = false;
                return;
            }

            //auto start = std::chrono::system_clock::now();
            {
                auto multi_output = model->process(frames);
                //here is possible case when 1 video end before others
                // so make named map for each object
                for (int idx = 0; idx < multi_output.size(); ++idx)
                    central_widget_.putTo(out_idxs[idx], frames[idx], multi_output[idx]);
            }
            //auto end = std::chrono::system_clock::now();
            //auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            //std::string msg = "Processed: " + std::to_string(int_ms) + " ms";
            //sendOutput(msg);
        }
    }
    catch (const std::string &) {
    }
}

void MainWindow::sendOutput(const std::string &message) {
    auto value_copy = std::string(message);
    central_widget_.sendOutput(message);
}
