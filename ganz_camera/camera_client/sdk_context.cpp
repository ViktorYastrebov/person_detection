#include "sdk_context.h"
#include "sdks.h"

#include <exception>

namespace ganz_camera {

    SDKContext::SDKContext() {
        int ret = sdks_dev_init(nullptr);
        if (ret) {
            throw std::exception("SDK initialization error occurs");
        }
    }

    SDKContext::~SDKContext() {
        sdks_dev_quit();
    }

}