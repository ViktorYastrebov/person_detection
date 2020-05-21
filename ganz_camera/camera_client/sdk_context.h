#pragma once

#include <memory>
#include "camera_client/connection.h"

namespace ganz_camera {

    // INFO: can be made as singleton
    class SDKContext  final {
    public:
        SDKContext();
        ~SDKContext();
        Connection& buildConnection(const std::string &host, const std::string &user, const std::string &pwd, bool ssl = false);
    private:
        std::unique_ptr<Connection> conn_;
    };
}