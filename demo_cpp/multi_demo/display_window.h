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
#include "trackers_pool.h"
#include <tbb/concurrent_queue.h>

//INFO: NEEDS REFACTORING !!!
//INFO: for split the functionality between show detections & tracking
//      use different classes; DisplayFrameDetections with tbb::concurent_queue<InputData>
//      AND DisplayFrameTracks with tbb::concurent_queue<InputTrackData>
//      + visualization differs

class DisplayFrame : public QFrame {
public:

#if 0
    struct InputData {
        cv::Mat frame;
        std::vector<DetectionResult> results;
    };
#endif

    struct InputTrackData {
        cv::Mat frame;
        std::vector<tracker::TrackResult> results;
    };

    DisplayFrame(QWidget *parent);
    ~DisplayFrame();

#if 0
    void put(const InputData& data);
#endif
    void put(const InputTrackData &dat);
    void stop();

protected:
    void paintEvent(QPaintEvent *) override;
    void displayTracks(QPaintEvent *);
#if 0
    void displayDetections(QPaintEvent *);
#endif

private:
#if 0
    tbb::concurrent_queue<InputData> queue_;
#endif
    tbb::concurrent_queue<InputTrackData> queue_;
    std::atomic_bool stop_;
    std::array<int, 3> color_;
    cv::Mat last_;
};

//INFO: needs refactorying, Interface depend on the DisplayFrame Type( detection or tracking)
//      this is not funny !!!
class ViewsWidget : public QWidget {
public:
    ViewsWidget(QWidget *parent);
    ~ViewsWidget();

    //void putTo(int idx, cv::Mat frame, const std::vector<DetectionResult> &res);
    void putTo(int idx, cv::Mat frame, const std::vector<tracker::TrackResult> &retults);
    void stopView(int idx);

private:
    QGridLayout* layout_;
    std::vector<DisplayFrame *> views_;
};


//INFO: needs refactorying, Interface depend on the DisplayFrame Type( detection or tracking)
//      this is not funny !!!
class CentralWidget : public QWidget {
//    Q_OBJECT
public:
    CentralWidget(QWidget *parent);
    ~CentralWidget();

#if 0
    void putTo(int idx, cv::Mat frame, const std::vector<DetectionResult> &res);
#endif
    void putTo(int idx, cv::Mat frame, const std::vector<tracker::TrackResult> &results);
    void stopView(int idx);
    void sendOutput(const std::string &msg);

private:
    QVBoxLayout *base_layout_;
    QTextEdit *output_;
    ViewsWidget *views_widget_;
};