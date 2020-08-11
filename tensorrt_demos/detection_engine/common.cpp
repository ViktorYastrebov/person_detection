#include "common.h"
#include <cuda_runtime_api.h>

namespace common {

    DeviceBuffers::DeviceBuffers(const int input_size, const int output_size)
    {
        cudaMalloc(&buffers[0], input_size);
        cudaMalloc(&buffers[1], output_size);
    }
    DeviceBuffers::~DeviceBuffers() {
        cudaFree(&buffers[0]);
        cudaFree(&buffers[1]);
    }

    void *DeviceBuffers::getBuffer(const BUFFER_TYPE idx) {
        return buffers[idx];
    }

    void **DeviceBuffers::getAll() {
        return buffers;
    }
}