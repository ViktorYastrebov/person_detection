#pragma once

#include "connection.h"

#include "alarm_data.h"

namespace ganz_camera {

    namespace callback_wrapper {
        void alarm_handler(unsigned int handle, void** p_data, void* p_obj);
    }

    class GANZ_CAMERA_ENGINE_DECL AlarmHandler {
    public:
        AlarmHandler(Connection &owner);
        ~AlarmHandler();
    private:
        friend void callback_wrapper::alarm_handler(unsigned int handle, void** p_data, void* p_obj);

        void handle(const AlarmData &alarm);

        Connection &owner_;
    };
}