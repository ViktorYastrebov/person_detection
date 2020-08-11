#pragma once 

#include <memory>

namespace common {
    struct InferDeleter {
        template <typename T>
        void operator()(T* obj) const {
            if (obj) {
                obj->destroy();
            }
        }
    };

    template<class T>
    using TensorRTUPtr = std::unique_ptr<T, common::InferDeleter>;
}