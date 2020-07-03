#pragma once
#include <string>
#include <vector>
#include <map>
#include "device_utils.h"

class BaseModelBatched;

struct ModelBuilder {
    ModelBuilder(const std::string &name, const std::string &conf);
    std::unique_ptr<BaseModelBatched> build(const std::vector<std::string> &files, const std::vector<int> &classes, RUN_ON on) const;
private:
    std::string name_;
    float conf_;
};