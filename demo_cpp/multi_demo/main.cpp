#include <qapplication.h>
#include <qcommandlineparser.h>
#include <qscreen.h>
#include "main_window.h"


int main(int argc, char*argv[]) {
    QApplication app(argc, argv);
    MainWindow win;

    constexpr const int MAX_FILES = 4;

    QCommandLineParser parser;
    parser.addHelpOption();
    //parser.addPositionalArgument("-n", "Set name of the network, should be \"YoloV3\" or \"YoloV4\"");
    //parser.addPositionalArgument("-c", "Confidence threshold [0.0, 1.0]");

    QCommandLineOption modelNameOption(QStringList() << "n" << "name", "Set <name> of the network, should be \"YoloV3\" or \"YoloV4\"", "<name>", "YoloV3");
    parser.addOption(modelNameOption);
    QCommandLineOption confidenceOption("c", "Confidence threshold[0.0, 1.0]", "<conf>", "0.3");
    parser.addOption(confidenceOption);
    QCommandLineOption filesOption(QStringList() << "f" << "file", 
                            "Video files",
                        "<file path>");
    parser.addOption(filesOption);

    parser.process(app);

    std::string model_name = parser.value(modelNameOption).toStdString();
    std::string model_conf = parser.value(confidenceOption).toStdString();
    std::vector<std::string> inputs;

    auto qfiles = parser.values(filesOption);
    for (const auto &f : qfiles) {
        inputs.push_back(f.toStdString());
    }
    win.Process(model_name, model_conf, inputs);

    QScreen *screen = app.primaryScreen();
    QRect rect = screen->availableGeometry();
    win.setGeometry(rect);

    win.show();
    return app.exec();
}