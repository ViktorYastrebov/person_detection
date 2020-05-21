#include "connection.h"
#include <stdio.h>
#include <windows.h>
#include "sdks.h"

namespace ganz_camera {

    namespace callback_wrapper {
        void disconnect_handler(unsigned int handle, void* p_obj) {
            Connection *owner = static_cast<Connection*>(p_obj);
            owner->disconnect_handler();
        }
    }


    Connection::Connection(const std::string &host, const std::string &user, const std::string &pwd, bool ssl)
        : host_(host)
        , user_(user)
        , pwd_(pwd)
        , port_(ssl ? 20001 : 30001)
        , conn_handle_(0) // invalid by default
    {
        if (ssl) {
            conn_handle_ = sdks_dev_conn_ssl(host_.c_str(),
                port_,
                user_.c_str(),
                pwd_.c_str(),
                callback_wrapper::disconnect_handler,
                this);
        } else {
            conn_handle_ = sdks_dev_conn(host_.c_str(),
                port_,
                user_.c_str(),
                pwd_.c_str(),
                callback_wrapper::disconnect_handler,
                this);
        }

        if (!conn_handle_) {
            std::string error = "Can't connect to server : " + host + ":" + std::to_string(port_);
            throw std::exception(error.c_str());
        }
    }

    Connection::~Connection()
    {
        if (conn_handle_) {
            sdks_dev_conn_close(conn_handle_);
        }
    }

    unsigned int Connection::getHandle() const {
        return conn_handle_;
    }

    void Connection::disconnect_handler() {
        //INFO: can be used for notifications or data releasing
    }
}