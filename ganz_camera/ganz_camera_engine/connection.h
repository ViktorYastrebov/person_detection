#pragma once

#include <string>
#include "decl_spec.h"

#pragma warning(push)
#pragma warning(disable: 4251)

namespace ganz_camera {

    namespace callback_wrapper {
        void disconnect_handler(unsigned int handle, void* p_obj);
    }

    enum GANZ_CAMERA_ENGINE_DECL STREAM_TYPE : int {
        HD = 1,
        SD = 2
        //NOT_SUPPORTED_SMOOTH = 3
    };

    class GANZ_CAMERA_ENGINE_DECL Connection final {
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

#pragma warning(pop)