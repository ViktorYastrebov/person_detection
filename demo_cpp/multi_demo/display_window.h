#pragma once

#include <qwidget.h>
#include <qframe.h>
#include <qgridlayout.h>
#include <qpainter.h>

#include <array>


class DisplayFrame : public QFrame {
//    Q_OBJECT
public:
    DisplayFrame();
    ~DisplayFrame();

protected:
    void paintEvent(QPaintEvent *) override;
private:
  //  QPainter painter;
    std::array<int, 3> color_;
};

class CentralWidget : public QWidget {
//    Q_OBJECT
public:
    CentralWidget(QWidget *parent);
    ~CentralWidget();
private:
    QGridLayout* layout_;
    std::vector<DisplayFrame *> frames_;
};