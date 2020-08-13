#include "common.h"
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace common {

    DeviceBuffers::DeviceBuffers(const int input_size, const int output_size)
    {
        cudaError_t error = cudaMalloc(&buffers[BUFFER_TYPE::INPUT], input_size);
        if (error != cudaSuccess) {
            throw std::runtime_error("Can't allocate device input buffer memory");
        }
        error = cudaMalloc(&buffers[BUFFER_TYPE::OUT], output_size);
        if (error != cudaSuccess) {
            throw std::runtime_error("Can't allocate device output buffer memory");
        }
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

    HostBuffers::HostBuffers(const int input_size, const int output_size) {
        cudaError_t error = cudaHostAlloc(&buffers[BUFFER_TYPE::INPUT], input_size, cudaHostAllocPortable);
        if (error != cudaSuccess) {
            throw std::runtime_error("Can't allocate host input buffer memory");
        }
        error = cudaHostAlloc(&buffers[BUFFER_TYPE::OUT], output_size, cudaHostAllocPortable);
        if (error != cudaSuccess) {
            throw std::runtime_error("Can't allocate host output buffer memory");
        }
    }
    HostBuffers::~HostBuffers() {
        cudaFree(buffers[BUFFER_TYPE::INPUT]);
        cudaFree(buffers[BUFFER_TYPE::OUT]);
    }

    void *HostBuffers::getBuffer(const BUFFER_TYPE idx) {
        return buffers[idx];
    }

    void **HostBuffers::getAll() {
        return buffers;
    }
}