#include "display_window.h"
#include <qgridlayout.h>


DisplayFrame::DisplayFrame()
    :QFrame()
{
    for (int i = 0; i < 3; ++i) {
        color_[i] = rand() % 255;
    }
    
}

DisplayFrame::~DisplayFrame()
{}

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
            frames_.push_back(frame);
            layout_->addWidget(frame, i, j);
        }
    }
    setLayout(layout_);
}

CentralWidget::~CentralWidget()
{
    for (const auto w : frames_) {
        delete w;
    }
    frames_.clear();
    delete layout_;
}
