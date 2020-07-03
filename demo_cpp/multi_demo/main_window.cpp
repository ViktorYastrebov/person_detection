#include "main_window.h"

MainWindow::MainWindow()
    :QMainWindow()
    , central_widget_(this)
{
    setCentralWidget(&central_widget_);
}

MainWindow::~MainWindow()
{}


void MainWindow::Process(const std::string &name, const std::string &conf, const std::vector<std::string> &files) {

}

CentralWidget &MainWindow::getCentralWidget() {
    return central_widget_;
}

const CentralWidget &MainWindow::getCentralWidget() const {
    return central_widget_;
}

