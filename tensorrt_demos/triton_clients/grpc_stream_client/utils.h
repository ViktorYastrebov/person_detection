#include "common/datatypes.h"
#include <opencv2/core.hpp>

#include <vector>
#include <memory>
#include <string>

#include "cuda_buffer.h"

namespace triton_inference {
    class GRPCClient {
    public:
        GRPCClient(const std::string &url, const std::string &model_name, const int BATCH_SIZE, const std::vector<int> &classes_ids, bool verbose = false);
        ~GRPCClient();
        std::vector<common::datatypes::DetectionResults> inference(const std::vector<cv::Mat> &imagesRGB, const float confidence, const float nms_threshold);

        void print_stats();

    private:
        //TODO: make it general and part of the yolov5 common shared DLL
        std::vector<cv::Mat> preprocessImage(const std::vector<cv::Mat> &imgs);
        void prepareBuffer(const std::vector<cv::Mat> &preparedImgs);
        std::vector<common::datatypes::DetectionResults> processResults(float *results, const std::vector<cv::Mat> &oginalImgs, const float conf, const float nms_thresh);

    private:
        std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> client_;
        std::shared_ptr<nvidia::inferenceserver::client::InferInput> input_;
        std::shared_ptr<nvidia::inferenceserver::client::InferRequestedOutput> output_;
        std::vector<int> classes_ids_;
        nvidia::inferenceserver::client::InferOptions options_;

        std::unique_ptr<InputCudaBuffer> input_cuda_buffer_;
        std::unique_ptr<OutputCudaBuffer> output_cuda_buffer_;

        float *input_data_;
        float *output_data_;

        int INPUT_DATA_SIZE;
        int OUTPUT_DATA_SIZE;
        int BATCH_SIZE_;
        std::shared_ptr<nvidia::inferenceserver::client::InferResult> results_ptr;
    };
}
