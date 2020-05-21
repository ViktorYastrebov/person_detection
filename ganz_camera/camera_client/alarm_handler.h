#pragma once

#include "connection.h"

#include "json/json.hpp"

namespace ganz_camera {

    namespace callback_wrapper {
        void alarm_handler(unsigned int handle, void** p_data, void* p_obj);
    }

    class AlarmHandler {
    public:
        AlarmHandler(Connection &owner);
        ~AlarmHandler();
    private:
        friend void callback_wrapper::alarm_handler(unsigned int handle, void** p_data, void* p_obj);

        void handle(const nlohmann::json &json);

        Connection &owner_;
    };
}