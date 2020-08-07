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


    void ENGINE_DECL test_generate_grid(int x_dim, int y_dim);
    void ENGINE_DECL test_normalize_xy(int channels, int x_dim, int y_dim);
    void ENGINE_DECL test_normalize_wh(int channels, int x_dim, int y_dim);

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


        struct Grid {
            using key_t = std::tuple<int, int>;
            Grid(int x_size, int y_size);
            ~Grid();
            key_t key() const;
            float *deviceMemory();
        private:
            int x_size_;
            int y_size_;
            float *generated_grid_;
        };

        struct DetectionNode {
            void process(float *gpu_tensort, int number_anchors, int num_x, int num_y, int num_out, const float stride, const float achors[3][2]
                //,cudaStream_t cudaStream
            );
            using GridMapType = std::map<Grid::key_t, std::shared_ptr<Grid>>;
            GridMapType grids_;
        };

        const float strides[3] = { 8.0f, 16.0f, 32.0f };
        const float yolov3_spp_anchors[3][3][2] = {
            { {10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f} },
            { {30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f} },
            { {116.0f, 90.0f}, {156.0f,198.0f}, {373.0f, 326.0f} }
        };

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
