#pragma once

#include "base_model.h"

#include "common/logging.h"
#include "common/common.h"

#include "NvInfer.h"
#include "NvInferRuntime.h"

namespace detector {
#pragma warning(push)
#pragma warning(disable: 4251)
    class ENGINE_DECL CommonDetector : public BaseDetector {
    public:
        virtual ~CommonDetector() = default;
        virtual common::datatypes::DetectionResults inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold) override;
    protected:

        Logger gLogger_;
        int batch_size_ = 1;
        std::vector<char> deserialized_buffer_;
        common::TensorRTUPtr<nvinfer1::IRuntime> runtime_;
        common::TensorRTUPtr<nvinfer1::ICudaEngine> engine_;
        common::TensorRTUPtr<nvinfer1::IExecutionContext> context_;
        std::unique_ptr<common::HostBuffers> host_buffers_;
        std::unique_ptr<common::DeviceBuffers> device_buffers_;
        cudaStream_t stream_;
    };
#pragma warning(pop)
}