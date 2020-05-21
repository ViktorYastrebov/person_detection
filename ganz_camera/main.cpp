#include <iostream>
#include <string>

#include "json/json.hpp"

#include "camera_client/sdk_context.h"
#include "camera_client/face_handler.h"
#include "camera_client/video_stream.h"

int main(int argc, char *argv[])
{
    std::string host;
    unsigned short port = 0;
    std::string user;
    std::string pwd;
    bool is_ssl = false;

    if (argc != 5) {
        std::cout << "Usage: server_demo.exe ssl/no-ssl IP user password";
        return 0;
    } else {
        std::string test_ssl(argv[1]);
        is_ssl = test_ssl == "ssl";
        host = std::string(argv[2]);
        // port = std::atoi(argv[3]);
        user = std::string(argv[3]);
        pwd = std::string(argv[4]);
    }

    ganz_camera::SDKContext context;
    ganz_camera::Connection & connection = context.buildConnection(host, user, pwd, is_ssl);

    ganz_camera::FaceHandler faceHandler(connection);
    ganz_camera::VideoStream stream(connection, 1, ganz_camera::VideoStream::HD);

    //SOME INFINIT LOOP
    std::string input;
    while (std::getline(std::cin, input)) {
        if (input == "exit") {
            break;
        }
    }

    return 0;
}