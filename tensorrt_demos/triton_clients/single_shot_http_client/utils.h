#include "http_client.h"
#include <memory>
#include <string>
#include "common/datatypes.h"
#include <opencv2/core.hpp>
#include <vector>

//namespace ni = nvidia::inferenceserver;
//namespace nic = nvidia::inferenceserver::client;

namespace triton_inference {
    class Client {
    public:
        Client(const std::string &url, const std::string &model_name,  const std::vector<int> &classes_ids, bool verbose = false);
        ~Client();
        common::datatypes::DetectionResults inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold);

    private:
        //TODO: make it general and part of the yolov5 common shared DLL
        cv::Mat preprocessImage(const cv::Mat &img);
        void prepareBuffer(const cv::Mat &prepared);
        common::datatypes::DetectionResults processResults(float *results, const cv::Mat &prepared, const float conf, const float nms_thresh);

    private:
        std::unique_ptr<nvidia::inferenceserver::client::InferenceServerHttpClient> client_;
        std::shared_ptr<nvidia::inferenceserver::client::InferInput> input_;
        //nvidia::inferenceserver::client::InferInput* input;
        std::shared_ptr<nvidia::inferenceserver::client::InferRequestedOutput> output_;
        std::vector<int> classes_ids_;
        nvidia::inferenceserver::client::InferOptions options_;
        float *input_data_;
        float * output_data_;

        int INPUT_DATA_SIZE;
        int OUTPUT_DATA_SIZE;

    };
}
