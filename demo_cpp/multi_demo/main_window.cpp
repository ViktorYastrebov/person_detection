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
{}

void MainWindow::Process(const std::string &name, const std::string &conf, const std::vector<std::string> &files) {
    auto func = std::bind(&MainWindow::ProcessImpl, this, name, conf, files);
    runner_ = std::thread(func);
}

void MainWindow::ProcessImpl(const std::string &name, const std::string &conf, const std::vector<std::string> &files) {
    try {
        ModelBuilder builder(name, conf);
        auto model = builder.build(files, { 0 }, RUN_ON::GPU);

        std::vector<cv::VideoCapture> streams;
        std::unordered_map<cv::VideoCapture*, int> vid2ui;
        int idx = 0;
        for (const auto &file : files) {
            auto stream = cv::VideoCapture(file);
            streams.push_back(stream);
            vid2ui[&stream] = idx;
            ++idx;
        }

        bool processing = true;
        while (processing) {
            std::vector<cv::Mat> frames;
            std::vector<int> out_idxs;
            for (auto &s : streams) {
                cv::Mat frame;
                if (s.read(frame)) {
                    frames.push_back(frame);
                    out_idxs.push_back(vid2ui.at(&s));
                } else {
                    central_widget_.stopView(vid2ui[&s]);
                }
            }
            if (out_idxs.empty()) {
                processing = false;
                return;
            }

            auto start = std::chrono::system_clock::now();
            {
                auto multi_output = model->process(frames);
                //here is possible case when 1 video end before others
                // so make named map for each object
                for (int idx = 0; idx < multi_output.size(); ++idx)
                    central_widget_.putTo(out_idxs[idx], frames[idx], multi_output[idx]);
            }
            auto end = std::chrono::system_clock::now();
            auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        }
    }
    catch (const std::string &) {
    }
}
