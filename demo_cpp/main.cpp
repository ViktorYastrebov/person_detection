#include <iostream>
#include <memory>

#include "argparser/argparser.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include "detection_engine/yolov3_model.h"

#include <opencv2/core/ocl.hpp>

void process_video_stream(const std::string &file, const std::string &weights, const std::string &model_config) {
    auto model = std::make_unique<YoloV3>(weights, model_config, RUN_ON::OPENCL);
    if (model) {
        cv::Mat frame;
        cv::VideoCapture stream(file);

        cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
        {
            while (stream.read(frame)) {
                auto bboxes = model->process(frame);
                std::cout << "Persons on the frame : " << bboxes.size() << std::endl;
                for (const auto &bbox : bboxes) {
                    cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2);
                }
                cv::imshow("result", frame);
                int key = cv::waitKey(1);
                if (key == 27) {
                    break;
                }
            }
        }
        cv::destroyAllWindows();
    }
}

void test_opencl_devices() {
    if (!cv::ocl::haveOpenCL())
    {
        std::cout << "OpenCL is not available..." << std::endl;
        return;
    }
    cv::ocl::setUseOpenCL(true);
    std::vector< cv::ocl::PlatformInfo > platform_info;
    cv::ocl::getPlatfomsInfo(platform_info);

    for (int i = 0; i < platform_info.size(); i++) {
        cv::ocl::PlatformInfo sdk = platform_info.at(i);
        for (int j = 0; j < sdk.deviceNumber(); j++) {
            cv::ocl::Device device;
            sdk.getDevice(device, j);

            std::cout << "\n\n*********************\n Device " << i + 1 << std::endl;
            std::cout << "Vendor ID: " << device.vendorID() << std::endl;
            std::cout << "Vendor name: " << device.vendorName() << std::endl;
            std::cout << "Name: " << device.name() << std::endl;
            std::cout << "Driver version: " << device.driverVersion() << std::endl;
            std::cout << "available: " << device.available() << std::endl;
        }
    }

    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
        std::cout << "Failed creating the context..." << std::endl;
        return;
    }

    std::cout << std::endl;
    std::cout << context.ndevices() << " GPU devices are detected." << std::endl;
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        std::cout << "name:              " << device.name() << std::endl;
        std::cout << "available:         " << device.available() << std::endl;
        std::cout << "imageSupport:      " << device.imageSupport() << std::endl;
        std::cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << std::endl;
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {
    ap::parser p(argc, argv);
    p.add("-f", "--file", "Path to video file", ap::mode::REQUIRED);
    p.add("-m", "--model", "Path to model weights", ap::mode::REQUIRED);
    p.add("-c", "--config", "Path to model config");
    p.add("-i", "--info", "List all OpenCL devices");

    auto args = p.parse();
    if (!args.parsed_successfully()) {
        return 0;
    }

    //INFO: just prints all the device list
    if (!args["-i"].empty()) {
        test_opencl_devices();
    }

    auto file = args["-f"];
    process_video_stream(file, args["-m"], args["-c"]);

    return 0;
}
