#pragma once

#include <qwidget.h>
#include <qmainwindow.h>
#include <qlayout.h>

#include <vector>
#include "display_window.h"
#include <thread>

class MainWindow : public QMainWindow {
//    Q_OBJECT
public:
    MainWindow();
    ~MainWindow();

    void Process(const std::string &name, const std::string &conf, const std::vector<std::string> &files);
protected:
    void ProcessImpl(const std::string &name, const std::string &conf, const std::vector<std::string> &files);

private:
    CentralWidget central_widget_;
    std::thread runner_;
};