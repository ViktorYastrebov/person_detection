#include "utils.h"

#include <opencv2/core/ocl.hpp>
#include <iostream>

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