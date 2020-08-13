#include "display_window.h"

#include <qfile.h>

DisplayFrame::DisplayFrame(QWidget *parent)
    :QFrame(parent)
    , queue_()
    , stop_(false)
    , last_frame_()
{
    for (int i = 0; i < 3; ++i) {
        color_[i] = rand() % 255;
    }
}

DisplayFrame::~DisplayFrame()
{}

void DisplayFrame::putInput(DisplayFrame::InputData &&data) {
    queue_.push(data);
}

void DisplayFrame::Stop() {
    stop_ = true;
}

void DisplayFrame::displayTracks(QPaintEvent *p) {
    if (!stop_) {
        InputData data;
        if (queue_.try_pop(data)) {
            auto frame = data.frame;

            QSize s = size();
            QRect rect = QRect(0, 0, s.width(), s.height());

            // Scale
            auto img_size = data.frame.size();

            float scaleWidth = static_cast<float>(rect.width()) / static_cast<float>(img_size.width);
            float scaleHeight = static_cast<float>(rect.height()) / static_cast<float>(img_size.height);
            float scale = std::min(scaleHeight, scaleWidth);

            auto new_width = static_cast<int>(img_size.width * scale);
            auto new_height = static_cast<int>(img_size.height *scale);

            for (auto &track : data.tracks_data) {
                track.bbox *= scale;
            }

            cv::Mat scaled;
            cv::resize(data.frame, scaled, cv::Size(new_width, new_height));
            QImage img = QImage((uchar*)scaled.data, scaled.cols, scaled.rows, scaled.step, QImage::Format_RGB888);

            QPixmap pixmapImage = QPixmap::fromImage(img);

            //draw everything on image
            {
                QPainter imagePainter(&pixmapImage);
                auto pen = QPen(QColor(0, 0, 255, 128));
                pen.setWidth(2);
                imagePainter.setPen(pen);
                for (const auto &track : data.tracks_data) {
                    imagePainter.drawRect(track.bbox(0), track.bbox(1), track.bbox(2), track.bbox(3));
                }
            }

            QPainter painter(this);

            auto w_diff = (rect.width() - new_width) / 2;
            rect.setX(rect.x() + w_diff);
            rect.setWidth(rect.width() - w_diff);

            auto h_diff = (rect.height() - new_height) / 2;
            rect.setY(rect.y() + h_diff);
            rect.setHeight(rect.height() - h_diff);

            auto result_image = pixmapImage.toImage();
            painter.drawImage(rect, result_image);
            last_frame_ = pixmapImage;
            QFrame::paintEvent(p);
            QFrame::update();
        } else {
            if (!last_frame_.isNull()) {
                QSize s = size();
                QRect rect = QRect(0, 0, s.width(), s.height());
                QPainter painter(this);
                auto result_image = last_frame_.toImage();
                painter.drawImage(rect, result_image);
                QFrame::paintEvent(p);
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
    }
}

void DisplayFrame::paintEvent(QPaintEvent *p) {
    displayTracks(p);
}
