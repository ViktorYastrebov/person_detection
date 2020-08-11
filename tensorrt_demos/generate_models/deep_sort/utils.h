#pragma once

#include <memory>

namespace helper {
    struct InferDeleter
    {
        template <typename T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };
    template<class T>
    using TensorRTPtr = std::unique_ptr<T, InferDeleter>;
}