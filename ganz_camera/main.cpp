#include <iostream>
#include <string>

#include "sdk_context.h"
#include "face_handler.h"

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

    ganz_camera::SDKContext context;
    {
        constexpr int channel = 1;
        ganz_camera::SDKContext::ConnectionPtr face_conn = context.buildConnection(host, user, pwd, is_ssl);
        ganz_camera::FaceHandler faceHandler(*face_conn, channel, ganz_camera::STREAM_TYPE::HD, ganz_camera::FaceHandler::BIG);

        std::cout << "Please type something" << std::endl;
        std::string input;
        std::getline(std::cin, input);
    }
    return 0;
}