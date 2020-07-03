#pragma once

#include <qwidget.h>
#include <qmainwindow.h>
#include <qlayout.h>

#include <vector>
#include "display_window.h"


class MainWindow : public QMainWindow {
//    Q_OBJECT
public:
    MainWindow();
    ~MainWindow();

    void Process(const std::string &name, const std::string &conf, const std::vector<std::string> &files);

    CentralWidget &getCentralWidget();
    const CentralWidget &getCentralWidget() const;

private:
    CentralWidget central_widget_;
};