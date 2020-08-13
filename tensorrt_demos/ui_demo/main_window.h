#pragma once

#include <qwidget.h>
#include <qmainwindow.h>
#include <qlayout.h>

#include <vector>
#include <thread>
#include <filesystem>


#include "display_window.h"

class MainWindow : public QMainWindow {
    //    Q_OBJECT
public:
    MainWindow(const std::filesystem::path &detector_path, const std::filesystem::path &deep_sort_path);
    ~MainWindow();

    void Process(const std::filesystem::path &video_file);

    //void sendOutput(const std::string &message);

protected:
    void ProcessImpl(const std::filesystem::path &file_path);
    void keyPressEvent(QKeyEvent *event) override;

private:
    std::filesystem::path detector_path_;
    std::filesystem::path deep_sort_path_;
    DisplayFrame display_frame_;
    std::thread runner_;
    std::atomic_bool stop_;
};