#pragma once 


#include "decl_spec.h"

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include <cuda_runtime_api.h>
#include "common/logging.h"
#include "common.h"

#include <opencv2/opencv.hpp>
#include <filesystem>

namespace detection_engine {

#pragma warning(push)
#pragma warning(disable: 4251)

    class ENGINE_DECL GenericDetector final {
    public:

        //TODO: make model dependent see generate_models code !!!
        static const int MAX_OUTPUT_COUNT = 80 * 80 + 40 * 40 + 20 * 20;
        static const int INPUT_H = 608;
        static const int INPUT_W = 608;
        static const int OUTPUT_SIZE = MAX_OUTPUT_COUNT * 7 + 1;  // we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1

        // static constexpr int LOCATIONS = 4;
        struct alignas(float) Detection {
            //x y w h
            float bbox[4];
            float det_confidence;
            float class_id;
            float class_confidence;
        };


        GenericDetector(const std::filesystem::path &model_path, const int BATCH_SIZE = 1);
        ~GenericDetector() = default;
        std::vector<cv::Rect> inference(const cv::Mat &imageRGB, const float confidence = 0.5, const float nms_threshold = 0.5);

    private:
        //INFO: temporary copy-past from adoption, might better use OpenCV NMS with AVX or even find implement GPU version
        struct Utils {
            static float iou(float lbox[4], float rbox[4]);
            static bool cmp(Detection& a, Detection& b);
            static void nms(std::vector<Detection>& res, float *output, const float conf, const float nms_thresh);
        };

        cv::Mat preprocessImage(const cv::Mat &imageRGB);
        void preapreBuffer(cv::Mat &prepared);
        std::vector<cv::Rect> processResults(const cv::Mat &prepared, const float conf, const float nms_thresh);

    private:

        static constexpr const char *INPUT_BLOB_NAME = "data";
        static constexpr const char *OUTPUT_BLOB_NAME = "prob";

        Logger gLogger_;
        int batch_size_;
        std::vector<char> deserialized_buffer_;
        float input_host_buffer_[3 * INPUT_H * INPUT_W];
        float output_host_buffer[OUTPUT_SIZE];
        common::TensorRTUPtr<nvinfer1::IRuntime> runtime_;
        common::TensorRTUPtr<nvinfer1::ICudaEngine> engine_;
        common::TensorRTUPtr<nvinfer1::IExecutionContext> context_;
        std::unique_ptr<common::DeviceBuffers> device_buffers_;
        cudaStream_t stream_;
    };
#pragma warning(pop)
}
