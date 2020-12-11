
#define TRITON_ENABLE_GPU 1
#include "grpc_client.h"
#include <cstddef>

namespace triton_inference {
    //TODO: make more suitable interface
    //     like updateData(), operator <<
    //     and pass reference to interfaces to setup buffers/call client methods
    struct InputCudaBuffer {
        InputCudaBuffer(std::size_t bytes,
                   std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> &client,
                   const std::string &name,
                   int device_id = 0);
        ~InputCudaBuffer();
        InputCudaBuffer(const InputCudaBuffer &) = delete;
        InputCudaBuffer & operator = (const InputCudaBuffer &) = delete;
        //void updateData(void *data, std::size_t bytes);
        void *getInternalBuffer();

        cudaIpcMemHandle_t getIPCHandle() const;
        std::size_t size() const;
    private:
        void *buffer_;
        std::size_t bytes_;
        std::string name_;
        std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> &client_;
        //INFO: defined by ipc.h in triton client built with GPU support
        cudaIpcMemHandle_t handle_;
    };

    struct OutputCudaBuffer {
        OutputCudaBuffer(std::size_t bytes, std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> &client,
                         const std::string &name,
                         int device_id = 0);
        ~OutputCudaBuffer();

        void *getInternalBuffer();
        cudaIpcMemHandle_t getIPCHandle() const;
        std::size_t size() const;

        private:
            std::size_t bytes_;
            std::string name_;
            std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> &client_;
            void *buffer_;
            cudaIpcMemHandle_t output_cuda_handle_;
    };

}
