#pragma once

#include <string>

namespace ganz_camera {

    namespace callback_wrapper {
        void disconnect_handler(unsigned int handle, void* p_obj);
    }

    class Connection final {
    public:
        Connection(const std::string &host, const std::string &user, const std::string &pwd, bool ssl = false);
        ~Connection();

        Connection(const Connection &) = delete;
        Connection & operator = (const Connection &) = delete;
        unsigned int getHandle() const;

    private:
        friend void callback_wrapper::disconnect_handler(unsigned int handle, void* p_obj);
        void disconnect_handler();

        std::string host_;
        std::string user_;
        std::string pwd_;
        unsigned short port_;
        unsigned int conn_handle_;
    };
}