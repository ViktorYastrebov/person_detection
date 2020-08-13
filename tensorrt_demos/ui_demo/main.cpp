#include <filesystem>
#include <qapplication.h>
#include <qcommandlineparser.h>
#include <qscreen.h>
#include "main_window.h"



int main(int argc, char*argv[]) {
    QApplication app(argc, argv);

    QCommandLineParser parser;
    parser.addHelpOption();

    QCommandLineOption detector_model_path("d", "detector model path", "<detector>");
    parser.addOption(detector_model_path);
    QCommandLineOption input_file_option("f", "input file name", "<input>");
    parser.addOption(input_file_option);

    QCommandLineOption deep_sort_model("s", "deep sort model", "<sort>");
    parser.addOption(deep_sort_model);

    //QCommandLineOption confidenceOption("c", "Confidence threshold[0.0, 1.0]", "<conf>", "0.3");
    //parser.addOption(confidenceOption);

    parser.process(app);

    auto detector_path = std::filesystem::path(parser.value(detector_model_path).toStdString());
    auto deep_sort_path = std::filesystem::path(parser.value(deep_sort_model).toStdString());
    auto input_file = std::filesystem::path(parser.value(input_file_option).toStdString());

    MainWindow win(detector_path, deep_sort_path);
    win.Process(input_file);

    QScreen *screen = app.primaryScreen();
    QRect rect = screen->availableGeometry();
    win.setGeometry(rect);

    win.show();
    return app.exec();
}