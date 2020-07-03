#include "display_window.h"
#include <qgridlayout.h>


DisplayFrame::DisplayFrame()
    :QFrame()
    , queue_()
    , stop_(false)
{
    for (int i = 0; i < 3; ++i) {
        color_[i] = rand() % 255;
    }
}

DisplayFrame::~DisplayFrame()
{}

void DisplayFrame::put(const InputData& data) {
    queue_.push(data);
}

void DisplayFrame::stop() {
    stop_ = true;
}

void DisplayFrame::paintEvent(QPaintEvent *p) {
    QSize s = size();
    QRect rect = QRect(0, 0, s.width(), s.height());
    QPainter painter(this);
    painter.fillRect(rect, QBrush(QColor(color_[0], color_[1], color_[2])));

    QFrame::paintEvent(p);
    QFrame::update();
}

CentralWidget::CentralWidget(QWidget *parent)
    :QWidget(parent)
    , layout_(new QGridLayout(this))
{
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            DisplayFrame *frame = new DisplayFrame();
            views_.push_back(frame);
            layout_->addWidget(frame, i, j);
        }
    }
    setLayout(layout_);
}

CentralWidget::~CentralWidget()
{
    for (const auto w : views_) {
        delete w;
    }
    views_.clear();
    delete layout_;
}

void CentralWidget::putTo(int idx, cv::Mat frame, const std::vector<DetectionResult> &res) {
    if (idx >= views_.size()) {
        return;
    }
    views_[idx]->put({ frame, res });
}

void CentralWidget::stopView(int idx) {
    if (idx >= views_.size()) {
        return;
    }
    views_[idx]->stop();
}
