#include "cuda_buffer.h"
#include <cuda_runtime_api.h>


namespace helper {
    void CreateCUDAIPCHandle(cudaIpcMemHandle_t* cuda_handle, void* input_d_ptr, int device_id) {
        // Set the GPU device to the desired GPU
        cudaError_t result = cudaSetDevice(device_id);
        if (result != cudaSuccess) {
            std::string error_string =  std::string("CreateCUDAIPCHandle, cudaSetDevice has failed with device_id = ") + std::to_string(device_id);
            throw std::runtime_error(error_string);
        }
        result = cudaIpcGetMemHandle(cuda_handle, input_d_ptr);
        if (result != cudaSuccess) {
            std::string error_string =  std::string("cudaIpcGetMemHandle has failed, error :") + std::string(cudaGetErrorName(result));
            throw std::runtime_error(error_string);
        }
    }
}

namespace triton_inference {
    InputCudaBuffer::InputCudaBuffer (std::size_t bytes, std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> &client, const std::string &name, int device_id)
        : buffer_()
        , bytes_(bytes)
        , name_(name)
        , client_(client)
    {
        cudaMalloc(&buffer_, bytes_);
        helper::CreateCUDAIPCHandle(&handle_, buffer_, device_id);

        auto error = client_->RegisterCudaSharedMemory(
                        name_,
                        handle_, device_id,
                        bytes_);
        if(!error.IsOk()) {
            std::string str_error = std::string("InputCudaBuffer failed : ") + error.Message();
            throw std::runtime_error(str_error);
        }
    }

    //might need to pass the std::funciton<> handler for errors
    InputCudaBuffer::~InputCudaBuffer()
    {
        auto error = client_->UnregisterCudaSharedMemory(name_);
        cudaError_t result = cudaFree(buffer_);
    }


    void *InputCudaBuffer::getInternalBuffer() {
        return buffer_;
    }

    cudaIpcMemHandle_t InputCudaBuffer::getIPCHandle() const {
        return handle_;
    }

    std::size_t InputCudaBuffer::size() const {
        return bytes_;
    }


    ///////////////////////////////////////////////

    OutputCudaBuffer::OutputCudaBuffer(std::size_t bytes,
                                       std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> &client,
                                       const std::string &name,
                                       int device_id)
        :bytes_(bytes)
        , name_(name)
        , client_(client)
    {
        cudaMalloc(&buffer_, bytes_);
        helper::CreateCUDAIPCHandle(&output_cuda_handle_, buffer_, device_id);

        auto error = client->RegisterCudaSharedMemory(
                name_, output_cuda_handle_, device_id,
                bytes_);

        if(!error.IsOk()) {
            std::string str_error = std::string("OutputCudaBuffer fialed : ") + error.Message();
            throw std::runtime_error(str_error);
        }
    }

    OutputCudaBuffer::~OutputCudaBuffer() {
        auto error = client_->UnregisterCudaSharedMemory(name_);
        cudaError_t result = cudaFree(buffer_);
    }

    void *OutputCudaBuffer::getInternalBuffer() {
        return buffer_;
    }

    cudaIpcMemHandle_t OutputCudaBuffer::getIPCHandle() const {
        return output_cuda_handle_;
    }

    std::size_t OutputCudaBuffer::size() const {
        return bytes_;
    }

}
