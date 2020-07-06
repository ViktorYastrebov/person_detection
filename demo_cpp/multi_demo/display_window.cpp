#include "display_window.h"
#include <qgridlayout.h>
#include <opencv2/imgproc.hpp>

#include <iostream>

DisplayFrame::DisplayFrame(QWidget *parent)
    :QFrame(parent)
    , queue_()
    , stop_(false)
    , last_()
{
    for (int i = 0; i < 3; ++i) {
        color_[i] = rand() % 255;
    }
}

DisplayFrame::~DisplayFrame()
{}

#if 0
void DisplayFrame::put(const InputData& data) {
    queue_.push(data);
}
#endif

void DisplayFrame::put(const InputTrackData &data) {
    queue_.push(data);
}

void DisplayFrame::stop() {
    stop_ = true;
}

#if 0
void DisplayFrame::displayDetections(QPaintEvent *p) {
    if (!stop_) {
        InputData data;
        if (queue_.try_pop(data)) {
            auto parent = parentWidget();
            auto central_widget = dynamic_cast<CentralWidget*>(parent);
            if (central_widget) {
                std::string msg = "Found detections : " + std::to_string(data.results.size());
                central_widget->sendOutput(msg);
            }
            for (const auto &rect : data.results) {
                cv::rectangle(data.frame, rect.bbox, cv::Scalar(0, 0, 255), 1);
            }
            QSize s = size();
            QRect rect = QRect(0, 0, s.width(), s.height());

            // Scale
            auto img_size = data.frame.size();

            float scaleWidth = static_cast<float>(rect.width()) / static_cast<float>(img_size.width);
            float scaleHeight = static_cast<float>(rect.height()) / static_cast<float>(img_size.height);
            float scale = std::min(scaleHeight, scaleWidth);

            auto new_width = static_cast<int>(img_size.width * scale);
            auto new_height = static_cast<int>(img_size.height *scale);

            cv::Mat scaled;
            cv::resize(data.frame, scaled, cv::Size(new_width, new_height));
            // end scale

            last_ = data.frame;

            //TODO: make investigation, does it need here or not at all !!!
            //cv::Mat forImage(scaled.cols, scaled.rows, scaled.type());
            //cv::cvtColor(scaled, forImage, cv::COLOR_RGB2BGR);

            //INFO: check memory managment because there is possible case when the Mat ref counter = 0 but QT still drawing for example
            //last_ = data.frame;
            QImage img = QImage((uchar*)scaled.data, scaled.cols, scaled.rows, scaled.step, QImage::Format_RGB888);
            QPainter painter(this);

            // Center the image in the frame
            auto w_diff = (rect.width() - new_width) / 2;
            rect.setX(rect.x() + w_diff);
            rect.setWidth(rect.width() - w_diff);

            auto h_diff = (rect.height() - new_height) / 2;
            rect.setY(rect.y() + h_diff);
            rect.setHeight(rect.height() - h_diff);

            painter.drawImage(rect, img);
            QFrame::paintEvent(p);
            //Might better to use rect only 
            QFrame::update();
        } else {
            if (!last_.empty()) {
                QPainter painter(this);
                QSize s = size();
                QRect rect = QRect(0, 0, s.width(), s.height());
                // Scale
                auto img_size = last_.size();

                float scaleWidth = static_cast<float>(rect.width()) / static_cast<float>(img_size.width);
                float scaleHeight = static_cast<float>(rect.height()) / static_cast<float>(img_size.height);
                float scale = std::min(scaleHeight, scaleWidth);

                auto new_width = static_cast<int>(img_size.width * scale);
                auto new_height = static_cast<int>(img_size.height *scale);

                cv::Mat scaled;
                cv::resize(last_, scaled, cv::Size(new_width, new_height));
                //cv::Mat forImage(scaled.cols, scaled.rows, scaled.type());
                //cv::cvtColor(scaled, forImage, cv::COLOR_RGB2BGR);

                QImage img = QImage((uchar*)scaled.data, scaled.cols, scaled.rows, scaled.step, QImage::Format_RGB888);

                auto w_diff = (rect.width() - new_width) / 2;
                rect.setX(rect.x() + w_diff);
                rect.setWidth(rect.width() - w_diff);

                auto h_diff = (rect.height() - new_height) / 2;
                rect.setY(rect.y() + h_diff);
                rect.setHeight(rect.height() - h_diff);

                painter.drawImage(rect, img);
                QFrame::paintEvent(p);
                //Might better to use rect only 
                QFrame::update();
            } else {
                QSize s = size();
                QRect rect = QRect(0, 0, s.width(), s.height());
                QPainter painter(this);
                painter.fillRect(rect, QBrush(QColor(color_[0], color_[1], color_[2])));
                QFrame::paintEvent(p);
                QFrame::update();
            }
        }
    } else {
        QSize s = size();
        QRect rect = QRect(0, 0, s.width(), s.height());
        QPainter painter(this);
        painter.fillRect(rect, QBrush(QColor(color_[0], color_[1], color_[2])));
        QFrame::paintEvent(p);
        QFrame::update();
    }
}
#endif

void DisplayFrame::displayTracks(QPaintEvent *p) {
    if (!stop_) {
        InputTrackData data;
        if (queue_.try_pop(data)) {
            auto parent = parentWidget();
            //auto central_widget = dynamic_cast<CentralWidget*>(parent);
            //if (central_widget) {
            //    std::string msg = "Found detections : " + std::to_string(data.results.size());
            //    central_widget->sendOutput(msg);
            //}
            auto class_color = cv::Scalar(0, 0, 255);
            auto overlay = data.frame.clone();
            for (const auto &result : data.results) {
                //auto class_color = classIdToColor[result.class_id];
                cv::rectangle(data.frame, result.bbox, class_color, 1);
                //std::string str = std::to_string(result.id) + " " + id_classes[result.class_id];
                std::string id_str = std::to_string(result.id);

                constexpr int TILE_HEIGHT = 10;
                cv::Rect tileRect;
                if ((result.bbox.y - TILE_HEIGHT) > 0) {
                    tileRect.x = result.bbox.x;
                    tileRect.y = result.bbox.y - TILE_HEIGHT;
                    tileRect.width = result.bbox.width;
                    tileRect.height = TILE_HEIGHT;
                } else {
                    tileRect.y = result.bbox.y + result.bbox.height;
                    tileRect.x = result.bbox.x;
                    tileRect.width = result.bbox.width;
                    tileRect.height = TILE_HEIGHT;
                }
                //INFO: It's not the best way to draw the labels, under OpenCV interface
                //      it's hard to fit the text to the box, so I leave the constants for now
                cv::rectangle(overlay, tileRect, class_color, cv::FILLED);
                cv::Point p(tileRect.x, tileRect.y + 5);
                cv::putText(data.frame, id_str, p, 0, 0.3, (127, 127, 127), 2); //VERY COST OP
            }
            auto alpha = 0.4;
            cv::addWeighted(overlay, alpha, data.frame, 1 - alpha, 0, data.frame);

            QSize s = size();
            QRect rect = QRect(0, 0, s.width(), s.height());

            // Scale
            auto img_size = data.frame.size();

            float scaleWidth = static_cast<float>(rect.width()) / static_cast<float>(img_size.width);
            float scaleHeight = static_cast<float>(rect.height()) / static_cast<float>(img_size.height);
            float scale = std::min(scaleHeight, scaleWidth);

            auto new_width = static_cast<int>(img_size.width * scale);
            auto new_height = static_cast<int>(img_size.height *scale);

            cv::Mat scaled;
            cv::resize(data.frame, scaled, cv::Size(new_width, new_height));
            // end scale

            last_ = data.frame;

            //TODO: make investigation, does it need here or not at all !!!
            //cv::Mat forImage(scaled.cols, scaled.rows, scaled.type());
            //cv::cvtColor(scaled, forImage, cv::COLOR_RGB2BGR);

            //INFO: check memory managment because there is possible case when the Mat ref counter = 0 but QT still drawing for example
            //last_ = data.frame;
            QImage img = QImage((uchar*)scaled.data, scaled.cols, scaled.rows, scaled.step, QImage::Format_RGB888);
            QPainter painter(this);

            // Center the image in the frame
            auto w_diff = (rect.width() - new_width) / 2;
            rect.setX(rect.x() + w_diff);
            rect.setWidth(rect.width() - w_diff);

            auto h_diff = (rect.height() - new_height) / 2;
            rect.setY(rect.y() + h_diff);
            rect.setHeight(rect.height() - h_diff);

            painter.drawImage(rect, img);
            QFrame::paintEvent(p);
            //Might better to use rect only 
            QFrame::update();
        } else {
            if (!last_.empty()) {
                QPainter painter(this);
                QSize s = size();
                QRect rect = QRect(0, 0, s.width(), s.height());
                // Scale
                auto img_size = last_.size();

                float scaleWidth = static_cast<float>(rect.width()) / static_cast<float>(img_size.width);
                float scaleHeight = static_cast<float>(rect.height()) / static_cast<float>(img_size.height);
                float scale = std::min(scaleHeight, scaleWidth);

                auto new_width = static_cast<int>(img_size.width * scale);
                auto new_height = static_cast<int>(img_size.height *scale);

                cv::Mat scaled;
                cv::resize(last_, scaled, cv::Size(new_width, new_height));
                //cv::Mat forImage(scaled.cols, scaled.rows, scaled.type());
                //cv::cvtColor(scaled, forImage, cv::COLOR_RGB2BGR);

                QImage img = QImage((uchar*)scaled.data, scaled.cols, scaled.rows, scaled.step, QImage::Format_RGB888);

                auto w_diff = (rect.width() - new_width) / 2;
                rect.setX(rect.x() + w_diff);
                rect.setWidth(rect.width() - w_diff);

                auto h_diff = (rect.height() - new_height) / 2;
                rect.setY(rect.y() + h_diff);
                rect.setHeight(rect.height() - h_diff);

                painter.drawImage(rect, img);
                QFrame::paintEvent(p);
                //Might better to use rect only 
                QFrame::update();
            } else {
                QSize s = size();
                QRect rect = QRect(0, 0, s.width(), s.height());
                QPainter painter(this);
                painter.fillRect(rect, QBrush(QColor(color_[0], color_[1], color_[2])));
                QFrame::paintEvent(p);
                QFrame::update();
            }
        }
    } else {
        QSize s = size();
        QRect rect = QRect(0, 0, s.width(), s.height());
        QPainter painter(this);
        painter.fillRect(rect, QBrush(QColor(color_[0], color_[1], color_[2])));
        QFrame::paintEvent(p);
        QFrame::update();
    }
}

void DisplayFrame::paintEvent(QPaintEvent *p) {
    displayTracks(p);
}


ViewsWidget::ViewsWidget(QWidget *parent)
    :QWidget(parent)
    , layout_(new QGridLayout(this))
{
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            DisplayFrame *frame = new DisplayFrame(parent);
            views_.push_back(frame);
            layout_->addWidget(frame, i, j);
        }
    }
    setLayout(layout_);
}

ViewsWidget::~ViewsWidget()
{
    for (const auto w : views_) {
        delete w;
    }
    views_.clear();
    delete layout_;
}

#if 0
void ViewsWidget::putTo(int idx, cv::Mat frame, const std::vector<DetectionResult> &res) {
    if (idx >= views_.size()) {
        return;
    }
    views_[idx]->put({ frame, res });
}
#endif

void ViewsWidget::putTo(int idx, cv::Mat frame, const std::vector<tracker::TrackResult> &retults) {
    if (idx >= views_.size()) {
        return;
    }
    //TODO: might need to avoid the copy, better use std::move, think about it 
    views_[idx]->put({ frame, retults });
}

void ViewsWidget::stopView(int idx) {
    if (idx >= views_.size()) {
        return;
    }
    views_[idx]->stop();
}


CentralWidget::CentralWidget(QWidget *parent)
    :QWidget(parent)
    , base_layout_(new QVBoxLayout(this))
{
    views_widget_ = new ViewsWidget(this);
    output_ = new QTextEdit();
    output_->setFocusPolicy(Qt::NoFocus);
    base_layout_->addWidget(views_widget_, 1);
    base_layout_->addWidget(output_, 0);
    setLayout(base_layout_);
}

CentralWidget::~CentralWidget()
{
    delete output_;
    delete base_layout_;
}

#if 0
void CentralWidget::putTo(int idx, cv::Mat frame, const std::vector<DetectionResult> &res) {
    views_widget_->putTo(idx, frame, res);
}
#endif

void CentralWidget::putTo(int idx, cv::Mat frame, const std::vector<tracker::TrackResult> &results) {
    views_widget_->putTo(idx, frame, results);
}

void CentralWidget::stopView(int idx) {
    views_widget_->stopView(idx);
}

void CentralWidget::sendOutput(const std::string &msg) {
    output_->append(QString(msg.c_str()));
}