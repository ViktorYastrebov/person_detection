#include <iostream>
#include <string>

#include "sdk_context.h"
#include "face_detector.h"

int main(int argc, char *argv[])
{
    std::string host;
    unsigned short port = 0;
    std::string user;
    std::string pwd;
    std::string str_port;
    bool is_ssl = false;

    if (argc != 5) {
        std::cout << "Usage: server_demo.exe ssl/no-ssl IP port user password";
        return 0;
    } else {
        std::string test_ssl(argv[1]);
        is_ssl = test_ssl == "ssl";
        host = std::string(argv[2]);
        user = std::string(argv[3]);
        pwd = std::string(argv[4]);
    }

    sunell_camera::SDKContext context;
    {
        constexpr int channel = 1;
        auto face_detector = context.createFaceDetector(host, user, pwd, is_ssl, 1, sunell_camera::STREAM_TYPE::SD, sunell_camera::PICTURE_SIZE::SMALL);

        std::cout << "Please type something" << std::endl;
        std::string input;
        std::getline(std::cin, input);
    }
    return 0;
}