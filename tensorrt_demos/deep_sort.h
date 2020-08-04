#pragma once 

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h" 
#include <cuda_runtime_api.h>
#include "logging.h"

#include <opencv2/opencv.hpp>

#include <filesystem>

namespace tensorrt_inference {
    struct InferDeleter
    {
        template <typename T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };

    class BufferManager;

    class DeepSortModel {
    public:

        template<class T>
        using TensorRTuniquePtr = std::unique_ptr<T, InferDeleter>;

        DeepSortModel(const std::string &path, const int BATCH_SIZE);
        DeepSortModel() = default;

        void setInputPath(const std::filesystem::path &path);

        bool prepareModel();
        bool inference();

    private:
        bool processInput(const BufferManager& buffers);
        bool verifyOutput(const BufferManager& buffers);
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
        //std::vector<cv::Mat> inputs_;
        std::filesystem::path input_path_;
        int images_processed_;
    };

}