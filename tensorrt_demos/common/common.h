#pragma once 

#include "decl_spec.h"
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
    using TensorRTUPtr = std::unique_ptr<T, InferDeleter>;

    enum BUFFER_TYPE : char {
        INPUT = 0,
        OUT = 1
    };

    struct COMMON_EXPORT DeviceBuffers {
        DeviceBuffers(const int input_size, const int output_size);
        ~DeviceBuffers();
        void *getBuffer(const BUFFER_TYPE idx);
        void **getAll();
    private:
        void* buffers[2];
    };

    struct COMMON_EXPORT HostBuffers {
        HostBuffers(const int input_size, const int output_size);
        ~HostBuffers();
        void *getBuffer(const BUFFER_TYPE idx);
        void **getAll();
    private:
        void* buffers[2];
    };

}