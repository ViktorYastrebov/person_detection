#include "generic_detector.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>

namespace detection_engine {
    namespace impl {
        __global__ void sigmoid(float *data) {
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            data[idx] = 1.0 / (1.0 + exp2f(-data[idx]));
        }

        //Generate on device
        // [ 0, 0
        //  1, 0
        //  N, 0
        //  ...
        //  0, 1
        //  1, 2
        //  N, N]
        __global__ void generate_grid(float *data) {
            int idx = blockIdx.x *blockDim.x + threadIdx.x * 2;
            data[idx] = threadIdx.x;
            data[idx + 1] = blockIdx.x;
        }

        __global__ void normalize_xy(float *data, float *grid, const int x_dim, const int y_dim, const float stride) {
            const int idx = (blockIdx.y * x_dim * y_dim +  blockIdx.x * blockDim.x + threadIdx.x) * 2;
            const int grid_id = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
#if 0
            //INFO: should give 0 in all cases
            data[idx] = data[idx] + grid[grid_id];
            data[idx + 1] = data[idx + 1] + grid[grid_id + 1];
#else
            data[idx] = (data[idx] * 2 - 0.5f + grid[grid_id]) * stride;
            data[idx + 1] = (data[idx + 1] * 2 - 0.5f + grid[grid_id + 1]) * stride;
#endif
        }

        // wheere y.shape = 1x3x80x80x85 and anchor.shape 1x3x1x1x2
        // which means. Apply each pair of 1/3 to 80x80 matrix
        //y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
        // shape offset is the 80x80*i so loop of 3 iteration covers all
        __global__ void normalize_wh(float *data, const int shape_offset, const float w_anchor, const float h_anchor, const int VECTOR_SIZE,
            const int start_idx, const int end_idx) {
            const int row_id = blockIdx.x * blockDim.x + threadIdx.x;
            const int idx = shape_offset + row_id * VECTOR_SIZE;
            const int w_idx = idx + start_idx;
            const int h_idx = idx + end_idx;
            data[w_idx] = data[w_idx] * data[w_idx] * w_anchor;
            data[h_idx] = data[h_idx] * data[h_idx] * h_anchor;
        }
    }

    void test_generate_grid(int x_dim, int y_dim) {
        float *device_buffer;
        cudaError_t error = cudaMalloc(&device_buffer, x_dim * y_dim * 2 * sizeof(float));
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        impl::generate_grid << < 10, 10 >> > (device_buffer);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            cudaFree(device_buffer);
            throw std::runtime_error(cudaGetErrorString(error));
        }
        cudaDeviceSynchronize();
#if DEBUG_CHECK
        float *check = new float[x_dim * y_dim * 2];
        cudaMemcpy(check, device_buffer, x_dim * y_dim * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "[";
        for (int i = 0; i < 100; ++i) {
            std::cout << check[i * 2] << ", " << check[2 * i + 1] << std::endl;
        }
#endif
        cudaFree(device_buffer);
    }

    void test_normalize_xy(int channels, int x_dim, int y_dim) {
        constexpr const std::size_t VECTOR_SIZE = 4;
        std::size_t total = channels * x_dim * y_dim * 2;
        float *test_data = new float[total];

        for (int i = 0; i < channels; ++i) {
            float data_test = 1.0f;
            for (int j = 0; j < y_dim * x_dim; ++j) {
                test_data[(i*y_dim * x_dim + j) * 2] = data_test;
                test_data[(i*y_dim * x_dim + j) * 2 + 1] = data_test + 1.0f;
                data_test += 2.0f;
            }
        }
        float data_test2 = 1.0f;
        float *test_data2 = new float[x_dim * y_dim * 2];
        for (int i = 0; i < x_dim * y_dim; ++i) {
            test_data2[i*2] = -data_test2;
            test_data2[i*2 + 1] = -data_test2 - 1.0f;
            data_test2 += 2.0f;
        }

        float *device_data;
        cudaError_t error = cudaMalloc(&device_data, total * sizeof(float));
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        float *device_data2;
        error = cudaMalloc(&device_data2, x_dim * y_dim * 2 * sizeof(float));
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

        error = cudaMemcpy(device_data, test_data, total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_data2, test_data2, x_dim * y_dim * 2 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(y_dim);
        dim3 gridDim(x_dim, channels);

        impl::normalize_xy << < gridDim, blockDim >> > (device_data, device_data2, x_dim, y_dim, 1.0f);
        error = cudaGetLastError();
        error = cudaDeviceSynchronize();
        error = cudaMemcpy(test_data, device_data, total * sizeof(float), cudaMemcpyDeviceToHost);
        //Check memory here
        cudaFree(device_data2);
        cudaFree(device_data);
        delete[] test_data2;
        delete[] test_data;
    }

    //INFO: test with vector size 8
    void test_normalize_wh(int channels, int x_dim, int y_dim) {
        constexpr const std::size_t VECTOR_SIZE = 8;
        auto total = channels * x_dim * y_dim;
        float * test_data = new float[total * VECTOR_SIZE];
        for (int k = 0; k < channels; ++k) {
            for (int i = 0; i < x_dim * y_dim; ++i) {
                float data = 1.0f;
                for (int j = 0; j < VECTOR_SIZE; ++j) {
                    test_data[k*x_dim * y_dim * VECTOR_SIZE + i * VECTOR_SIZE + j] = data;
                    data += 1.0f;
                }
            }
        }

        float *device_data;
        cudaError_t error = cudaMalloc(&device_data, total * VECTOR_SIZE * sizeof(float));
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

        error = cudaMemcpy(device_data, test_data, total * VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < channels; ++i) {
            float w_anchor = 0.1f;
            float h_anchor = 0.1f;
            impl::normalize_wh << < y_dim, x_dim >> > (device_data, y_dim * x_dim * VECTOR_SIZE *i, w_anchor, h_anchor, VECTOR_SIZE, 2, 3);
            error = cudaGetLastError();
        }

        error = cudaMemcpy(test_data, device_data, total * VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(device_data);
        for (int i = 0; i < total; ++i) {
            for (int j = 0; j < VECTOR_SIZE; ++j) {
                std::cout << test_data[i*total + j] << ", ";
            }
            std::cout << std::endl;
        }
        delete[] test_data;
    }


    GenericDetector::Grid::Grid(int x_size, int y_size)
        :x_size_(x_size)
        , y_size_(y_size_)
    {
        cudaError_t error = cudaMalloc(&generated_grid_, x_size_ * y_size_ * 2 * sizeof(float));
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        impl::generate_grid << < y_size_, x_size_ >> > (generated_grid_);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }

    GenericDetector::Grid::~Grid() {
        cudaFree(generated_grid_);
    }

    GenericDetector::Grid::key_t GenericDetector::Grid::key() const {
        return {x_size_, y_size_};
    }

    float * GenericDetector::Grid::deviceMemory() {
        return generated_grid_;
    }

    void GenericDetector::DetectionNode::process(float *gpu_tensort,
        int number_anchors, int num_x, int num_y, int num_out, const float stride, const float achors[3][2]
        //,cudaStream_t cudaStream
    )
    {
        //input: 1x3x80x80x85
        dim3 cthreads(number_anchors, num_x);
        dim3 cblocks(num_y, num_out);
        impl::sigmoid <<< cblocks, cthreads, 0 >>> (gpu_tensort);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        GridMapType::iterator it = grids_.find({ num_x, num_y });
        if (it == grids_.end()) {
            //DO NOT COPY, JUST KEEP THE DEVICE MEMORY ALLOCATED
            auto grid = std::make_shared<Grid>(num_x, num_y);
            auto ret = grids_.insert(std::make_pair(grid->key(), grid));
            it = ret.first;
        }
        // Tensor 3xNxK + 1xNxK just multiplies 3 times the same data
        dim3 blockDim(num_y);
        dim3 gridDim(num_x, number_anchors);
        impl::normalize_xy << < gridDim, blockDim >> > (gpu_tensort, it->second->deviceMemory(), num_x, num_y, stride);

        for (int i = 0; i < number_anchors; ++i) {
            impl::normalize_wh << < num_y, num_x>> > (gpu_tensort, num_y * num_x * i, achors[i][0], achors[i][1], num_out, 2, 3);
        }
    }

}