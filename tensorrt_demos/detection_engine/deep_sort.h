#pragma once

#include "decl_spec.h"

#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include <cuda_runtime_api.h>
#include "common/logging.h"
#include "common/common.h"
#include "common/datatypes.h"


namespace deep_sort_tracker {

#pragma warning(push)
#pragma warning(disable: 4251)

    class ENGINE_DECL DeepSort final {
    public:
        static constexpr const int MAX_BATCH_SIZE = 32;

        DeepSort(const std::filesystem::path &model_path, const int BATCH_SIZE = MAX_BATCH_SIZE);
        ~DeepSort() = default;
        common::datatypes::Detections getFeatures(const cv::Mat &imageRGB, const std::vector<common::datatypes::DetectionBox> &bboxes);

        static constexpr const int INPUT_W = 64;
        static constexpr const int INPUT_H = 128;
        static constexpr const int OUTPUT_SIZE = 512;
    private:
        void preapreBuffer(const std::vector<cv::Mat> &resized);
    private:
        Logger gLogger_;
        int batch_size_;
        std::vector<char> deserialized_buffer_;
        //float *input_host_buffer_;
        //float *output_host_buffer_;
        common::TensorRTUPtr<nvinfer1::IRuntime> runtime_;
        common::TensorRTUPtr<nvinfer1::ICudaEngine> engine_;
        common::TensorRTUPtr<nvinfer1::IExecutionContext> context_;
        std::unique_ptr<common::HostBuffers> host_buffers_;
        std::unique_ptr<common::DeviceBuffers> device_buffers_;
        cudaStream_t stream_;
    };
#pragma warning(pop)
}