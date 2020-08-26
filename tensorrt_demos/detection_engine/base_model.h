#pragma once

#include "decl_spec.h"

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include "common/datatypes.h"
#include "common/logging.h"
#include "common/common.h"

#include <opencv2/core.hpp>

namespace detector {

    class ENGINE_DECL BaseDetector {
    public:
        virtual ~BaseDetector();
        virtual common::datatypes::DetectionResults inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold) = 0;
    };

//INFO: can be implemented by Pimpl Idiom
//      class CommonDetector {
//      private:
//       struct Pimpl { //methods };
//       std::unique_ptr<Pimpl> pimpl_;
//      }
#pragma warning(push)
#pragma warning(disable: 4251)
    class ENGINE_DECL CommonDetector : public BaseDetector {
    public:
        virtual ~CommonDetector() = default;
        virtual common::datatypes::DetectionResults inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold) override;
    protected:
        Logger gLogger_;
        int batch_size_;
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