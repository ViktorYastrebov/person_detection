#pragma once

#include <qwidget.h>
#include <qframe.h>
#include <qgridlayout.h>
#include <QVBoxLayout>
#include <qpainter.h>
#include <qtextedit.h>
#include <tbb/concurrent_queue.h>

#include <array>
#include <opencv2/opencv.hpp>

#include "common/datatypes.h"

class DisplayFrame : public QFrame {
public:
    DisplayFrame(QWidget *parent);
    ~DisplayFrame();

    struct Entry {
        common::datatypes::DetectionBox bbox;
        int track_id;
        int class_id;
    };

    struct InputData {
        cv::Mat frame;
        std::vector<Entry> tracks_data;
    };

    void putInput(InputData &&data);
    void Stop();

protected:
    void paintEvent(QPaintEvent *) override;
    void displayTracks(QPaintEvent *);
private:
    tbb::concurrent_queue<InputData> queue_;
    std::atomic_bool stop_;
    std::array<int, 3> color_;
    //cv::Mat last_;
    //QPixmap last_frame_;
    QImage last_frame_;
};
