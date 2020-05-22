#pragma once

#include <memory>
#include "camera_client/connection.h"
#include <list>

namespace ganz_camera {

    // INFO: can be made as singleton
    class SDKContext  final {
    public:
        using ConnectionPtr = std::shared_ptr<Connection>;
        SDKContext();
        ~SDKContext();
        ConnectionPtr buildConnection(const std::string &host, const std::string &user, const std::string &pwd, bool ssl = false);
    private:
        std::list< ConnectionPtr > connections_;
    };
}