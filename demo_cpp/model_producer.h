#pragma once

#include <map>
#include <string>

#include <memory>
#include "device_utils.h"

//forward declaration
class BaseModel;

struct ModelProducer {
    ModelProducer() = default;
    ~ModelProducer() = default;
    std::unique_ptr< BaseModel > get(const std::string &model_name, RUN_ON on = RUN_ON::CPU);
};