#pragma once


#include "decl_spec.h"
#include "connection.h"
#include <memory>
#include <list>

#pragma warning(push)
#pragma warning(disable: 4251)

namespace ganz_camera {

    class GANZ_CAMERA_ENGINE_DECL SDKContext final {
    public:
        using ConnectionPtr = std::shared_ptr<Connection>;
        SDKContext();
        ~SDKContext();
        ConnectionPtr buildConnection(const std::string &host, const std::string &user, const std::string &pwd, bool ssl = false);
    private:
        std::list< ConnectionPtr > connections_;
    };
}

#pragma warning(pop)