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
    using TensorRTUPtr = std::unique_ptr<T, InferDeleter>;

    struct DeviceBuffers {
        enum BUFFER_TYPE : char {
            INPUT = 0,
            OUT = 1
        };
        DeviceBuffers(const int input_size, const int output_size);
        ~DeviceBuffers();
        void *getBuffer(const BUFFER_TYPE idx);
        void **getAll();
    private:
        void* buffers[2];
    };

}