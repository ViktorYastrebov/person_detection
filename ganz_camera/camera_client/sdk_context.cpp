#include "sdk_context.h"
#include "sdks.h"

#include <exception>

namespace ganz_camera {

    SDKContext::SDKContext()
        :connections_()
    {
        int ret = sdks_dev_init(nullptr);
        if (ret) {
            throw std::exception("SDK initialization error occurs");
        }
    }

    SDKContext::~SDKContext() {
        //INFO: order is important
        for (auto & con : connections_) {
            con.reset();
        }
        sdks_dev_quit();
    }

    SDKContext::ConnectionPtr SDKContext::buildConnection(const std::string &host, const std::string &user, const std::string &pwd, bool ssl) {
        auto connection = std::make_shared<Connection>(host, user, pwd, ssl);
        connections_.push_back(connection);
        return connection;
    }
}