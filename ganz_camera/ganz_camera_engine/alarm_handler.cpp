#include "alarm_handler.h"
#include "sdks.h"

#include <iostream>

namespace ganz_camera {

    namespace callback_wrapper {
        void alarm_handler(unsigned int handle, void** p_data, void* p_obj) {
            if (p_data) {
                ganz_camera::AlarmHandler* owner = static_cast<ganz_camera::AlarmHandler*>(p_obj);
                const char *data_ptr = static_cast<char*>(*p_data);
                std::cout << "alarm_handler : " << std::endl;
                std::cout << "data :" << data_ptr << std::endl;
                AlarmData data;
                data.fromJsonData(data_ptr);
                owner->handle(data);
            }
        }
    }

    AlarmHandler::AlarmHandler(Connection &owner)
        :owner_(owner)
    {
        int ret = sdks_dev_start_alarm(owner_.getHandle(), callback_wrapper::alarm_handler, this);
        if (ret) {
            throw std::exception("sdks_dev_start_alarm has failed");
        }
    }

    AlarmHandler::~AlarmHandler()
    {
        int ret = sdks_dev_stop_alarm(owner_.getHandle());
    }

    void AlarmHandler::handle(const AlarmData &alarm) {
        
    }

}