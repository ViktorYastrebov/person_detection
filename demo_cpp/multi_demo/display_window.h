#pragma once

#include <qwidget.h>
#include <qframe.h>
#include <qgridlayout.h>
#include <QVBoxLayout>
#include <qpainter.h>
#include <qtextedit.h>

#include <array>
#include <opencv2/core.hpp>
#include "base_model.h"
#include <tbb/concurrent_queue.h>

//INFO: NEEDS REFACTORING !!!
class DisplayFrame : public QFrame {
public:

    struct InputData {
        cv::Mat frame;
        std::vector<DetectionResult> results;
    };

    DisplayFrame(QWidget *parent);
    ~DisplayFrame();

    void put(const InputData& data);
    void stop();

protected:
    void paintEvent(QPaintEvent *) override;
private:
    tbb::concurrent_queue<InputData> queue_;
    std::atomic_bool stop_;
    std::array<int, 3> color_;
    cv::Mat last_;
};

class ViewsWidget : public QWidget {
public:
    ViewsWidget(QWidget *parent);
    ~ViewsWidget();

    void putTo(int idx, cv::Mat frame, const std::vector<DetectionResult> &res);
    void stopView(int idx);

private:
    QGridLayout* layout_;
    std::vector<DisplayFrame *> views_;
};


class CentralWidget : public QWidget {
//    Q_OBJECT
public:
    CentralWidget(QWidget *parent);
    ~CentralWidget();

    void putTo(int idx, cv::Mat frame, const std::vector<DetectionResult> &res);
    void stopView(int idx);
    void sendOutput(const std::string &msg);

private:
    QVBoxLayout *base_layout_;
    QTextEdit *output_;
    ViewsWidget *views_widget_;
};