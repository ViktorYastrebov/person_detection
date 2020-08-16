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
    last_frame_ = QImage();
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
                track.bbox = track.bbox * scale;
            }

            cv::Mat scaled;
            cv::resize(data.frame, scaled, cv::Size(new_width, new_height));
            QImage img = QImage((uchar*)scaled.data, scaled.cols, scaled.rows, scaled.step, QImage::Format_RGB888);

            //QPixmap pixmapImage = QPixmap::fromImage(img);

            //draw everything on image
            {
                QPainter imagePainter(&img);
                auto color = QColor(0, 0, 255, 128);
                auto pen = QPen(color);
                pen.setWidth(2);

                const auto &f = font();
                QFontMetrics metrics(f);
                int font_height = metrics.height();

                auto text_color = QColor(255, 255, 255);
                QPen text_pen = QPen(text_color);

                constexpr const int RECT_HEIGHT = 20;
                for (const auto &track : data.tracks_data) {

                    int x = static_cast<int>(track.bbox(0));
                    int y = static_cast<int>(track.bbox(1));
                    int w = static_cast<int>(track.bbox(2));
                    int h = static_cast<int>(track.bbox(3));
                    imagePainter.setPen(pen);
                    imagePainter.drawRect(x, y, w, h);
                    if (y - 2*font_height > 0) {
                        QRect tile_rect(x, y - 2*font_height, w, font_height*2);
                        imagePainter.fillRect(tile_rect, color);
                        std::string tile = "ID : " + std::to_string(track.track_id) + "\nClass ID :" + std::to_string(track.class_id);
                        imagePainter.setPen(text_pen);
                        imagePainter.drawText(tile_rect, QString::fromStdString(tile));
                    }
                }
            }

            QPainter painter(this);

            auto w_diff = (rect.width() - new_width) / 2;
            rect.setX(rect.x() + w_diff);
            rect.setWidth(rect.width() - w_diff);

            auto h_diff = (rect.height() - new_height) / 2;
            rect.setY(rect.y() + h_diff);
            rect.setHeight(rect.height() - h_diff);

            painter.drawImage(rect, img);
            last_frame_ = img.copy();
            QFrame::paintEvent(p);
            QFrame::update();
        } else {
            if (!last_frame_.isNull() && !stop_) {
                QPainter painter(this);
                QSize image_size = last_frame_.size();

                QSize s = size();
                QRect rect = QRect(0, 0, s.width(), s.height());

                auto w_diff = (rect.width() - image_size.width()) / 2;
                rect.setX(rect.x() + w_diff);
                rect.setWidth(rect.width() - w_diff);

                auto h_diff = (rect.height() - image_size.height()) / 2;
                rect.setY(rect.y() + h_diff);
                rect.setHeight(rect.height() - h_diff);

                painter.drawImage(rect, last_frame_);
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
