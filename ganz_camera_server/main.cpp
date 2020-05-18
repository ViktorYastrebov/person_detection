#include "server/server.h"
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "Help : " << std::endl;
        std::cout << "\trun like server_demo.exe host port," << std::endl;
        std::cout << "\tUsage example: server_demo.exe \"127.0.0.1\" 5555" << std::endl;
    }
    const std::string host(argv[1]);
    uint16_t port = static_cast<uint16_t>(std::atoi(argv[2]));
    try {
        HttpServer server(host, port, std::cout);
    }
    catch (const std::exception &ex) {
        std::cout << "Error occurs :" << ex.what() << std::endl;
    }
    return 0;
}