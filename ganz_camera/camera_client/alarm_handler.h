#pragma once

#include "connection.h"

#include "camera_client/alarm_data.h"

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

        void handle(const AlarmData &alarm);

        Connection &owner_;
    };
}