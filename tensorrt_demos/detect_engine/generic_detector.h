#pragma once 


#include "decl_spec.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h" 
#include <cuda_runtime_api.h>
#include "common/logging.h"
#include "common/common.h"

#include <opencv2/opencv.hpp>
#include <filesystem>


//forwards
namespace common {
    class BufferManager;
}

namespace detection_engine {

#pragma warning(push)
#pragma warning(disable: 4251)

    class ENGINE_DECL GenericDetector final {
    public:

        struct DetectionResult {
            std::array<int, 4> bbox;
            int class_id;
        };

        template<class T>
        using TensorRTuniquePtr = std::unique_ptr<T, common::InferDeleter>;

        GenericDetector(const std::string &path, const int BATCH_SIZE);
        ~GenericDetector();
        bool buildEngine();
        bool prepareBuffers();
        cv::Mat inference(const cv::Mat &imageRGB);

    private:
        struct InputInfo {
            cv::Mat input;
            float ratio;
            float dw, dh;
        };

        InputInfo preprocessImage(const cv::Mat &imageRGB);
        //DetectionResult


    private:
        std::string model_path_;
        int BATCH_SIZE;
        Logger gLogger_;
        TensorRTuniquePtr<nvinfer1::IBuilder> builder_;
        TensorRTuniquePtr<nvinfer1::INetworkDefinition> network_;
        TensorRTuniquePtr<nvinfer1::IBuilderConfig> config_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        nvinfer1::Dims input_dims_;
        nvinfer1::Dims output_dims_;
        std::unique_ptr<common::BufferManager> buffers_;
        TensorRTuniquePtr<nvinfer1::IExecutionContext> context_;
    };
#pragma warning(pop)
}
