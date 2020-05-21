#include "sdk_context.h"
#include "sdks.h"

#include <exception>

namespace ganz_camera {

    SDKContext::SDKContext()
        :conn_()
    {
        int ret = sdks_dev_init(nullptr);
        if (ret) {
            throw std::exception("SDK initialization error occurs");
        }
    }

    SDKContext::~SDKContext() {
        sdks_dev_quit();
    }

    Connection & SDKContext::buildConnection(const std::string &host, const std::string &user, const std::string &pwd, bool ssl) {
        conn_.reset(new Connection(host, user, pwd, ssl));
        return *conn_;
    }
}