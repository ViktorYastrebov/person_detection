#include "utils.h"
#include "common/yolov5/yolov5_layer.h"
#include <exception>
#include <stdexcept>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cuda_runtime_api.h>

namespace helpers_utils {
    float iou(float lbox[4], float rbox[4]) {
        float interBox[] = {
            std::max(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
            std::min(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
            std::max(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
            std::min(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
        };

        if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
        return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
    }

    bool cmp(YoloV5::Detection& a, YoloV5::Detection& b) {
        return a.conf > b.conf;
    }

    void nms(std::vector<YoloV5::Detection>& res, float *output, float conf_thresh, float nms_thresh, const std::vector<int> &filter_classes) {
        int det_size = sizeof(YoloV5::Detection) / sizeof(float);
        std::map<float, std::vector<YoloV5::Detection>> m;
        for (int i = 0; i < output[0] && i < YoloV5::MAX_OUTPUT_BBOX_COUNT; i++) {
            if (output[1 + det_size * i + YoloV5::LOCATIONS] <= conf_thresh) {
                continue;
            }

            //INFO: filter class at NMS time
            int class_id = static_cast<int>(output[2 + det_size * i + YoloV5::LOCATIONS]);
            auto it = std::find(filter_classes.cbegin(), filter_classes.cend(), class_id);
            if (it == filter_classes.cend()) {
                continue;
            }

            YoloV5::Detection det;
            memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
            if (m.count(det.class_id) == 0) {
                m.emplace(det.class_id, std::vector<YoloV5::Detection>());
            }
            m[det.class_id].push_back(det);
        }
        for (auto it = m.begin(); it != m.end(); it++) {
            auto& dets = it->second;
            std::sort(dets.begin(), dets.end(), cmp);
            for (size_t m = 0; m < dets.size(); ++m) {
                auto& item = dets[m];
                res.push_back(item);
                for (size_t n = m + 1; n < dets.size(); ++n) {
                    if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                        dets.erase(dets.begin() + n);
                        --n;
                    }
                }
            }
        }
    }
}


namespace triton_inference {
    namespace ni = nvidia::inferenceserver;
    namespace nic = nvidia::inferenceserver::client;
    using namespace YoloV5;

    GRPCClient::GRPCClient (const std::string &url, const std::string &model_name, const int BATCH_SIZE, const std::vector<int> &classes_ids,  bool verbose)
        : classes_ids_(classes_ids)
        , options_(model_name)
        , input_cuda_buffer_()
        , input_data_(nullptr)
        , output_data_(nullptr)
        , BATCH_SIZE_(BATCH_SIZE)
    {
        auto error = nic::InferenceServerGrpcClient::Create(&client_, url, verbose);
        if(!error.IsOk()) {
            throw std::runtime_error("Failed to create client");
        }

        error =  client_->UnregisterSystemSharedMemory();
        if(!error.IsOk()) {
            throw std::runtime_error("unable to unregister all system shared memory regions");
        }

        error =  client_->UnregisterCudaSharedMemory();
        if(!error.IsOk()) {
            throw std::runtime_error("unable to unregister all cuda shared memory regions");
        }

        std::vector<int64_t> shape{BATCH_SIZE, 3, INPUT_H, INPUT_W };
        // Initialize the inputs with the data.
        nic::InferInput* input;
        error = nic::InferInput::Create(&input, "data", shape, "FP32");
        if(!error.IsOk()) {
            throw std::runtime_error("unable to get input data");
        }
        input_.reset(input);

        //Milliseconds, please see common.h in triton_client/include
        // Client will wait for server processing
        constexpr const int CLIENT_TIMEOUT = 0;
        std::string model_version = "1";
        options_.model_version_ = model_version;
        options_.client_timeout_ = CLIENT_TIMEOUT;

        nic::InferRequestedOutput* output;
        error = nic::InferRequestedOutput::Create(&output, "prob");
        if(!error.IsOk()) {
            std::string error_str = std::string("unable to get 'prob', msg: ") + error.Message();
            throw std::runtime_error(error_str.c_str());
        }
        output_.reset(output);

        INPUT_DATA_SIZE = BATCH_SIZE_ * 3 * INPUT_H * INPUT_W;
        OUTPUT_DATA_SIZE = BATCH_SIZE_ * OUTPUT_SIZE;
        input_data_ = new float[INPUT_DATA_SIZE];
        output_data_ = new float[OUTPUT_DATA_SIZE];

        constexpr const char INPUT_BIND_NAME[] = "input_data";
        constexpr const int DEVICE_ID = 0;
        input_cuda_buffer_.reset(new InputCudaBuffer(INPUT_DATA_SIZE * sizeof(float), client_, INPUT_BIND_NAME, DEVICE_ID));
        error =input_->SetSharedMemory(INPUT_BIND_NAME, INPUT_DATA_SIZE * sizeof(float));
        if(!error.IsOk()) {
            std::string error_str = std::string("input SetSharedMemory failed : ") + error.Message();
            throw std::runtime_error(error_str.c_str());
        }

        constexpr const char OUTPUT_BIND_NAME[] = "output_data";
        output_cuda_buffer_.reset(new OutputCudaBuffer(OUTPUT_DATA_SIZE *sizeof(float), client_, OUTPUT_BIND_NAME, DEVICE_ID));
        error = output_->SetSharedMemory(OUTPUT_BIND_NAME, OUTPUT_DATA_SIZE *sizeof(float));
        if(!error.IsOk()) {
            std::string error_str = std::string("input SetSharedMemory failed : ") + error.Message();
            throw std::runtime_error(error_str.c_str());
        }
        //results_ptr.reset(new nic::InferResult);
    }

    GRPCClient::~GRPCClient() {
        if(input_data_) {
            delete[] input_data_;
            input_data_ = nullptr;
        }
        if(output_data_) {
            delete[] output_data_;
            output_data_ = nullptr;
        }
    }

    std::vector<common::datatypes::DetectionResults> GRPCClient::inference(const std::vector<cv::Mat> &imagesRGB, const float confidence, const float nms_threshold) {
        std::vector<cv::Mat> preparedImgs = preprocessImage(imagesRGB);
        prepareBuffer(preparedImgs);

        std::vector<nic::InferInput*> inputs{ input_.get() };
        std::vector<const nic::InferRequestedOutput*> outputs{ output_.get() };

        //TODO: might need to redo for multiple usage of inference()
        nic::InferResult* results;
        auto error =  client_->Infer(&results, options_, inputs, outputs);
        if(!error.IsOk()) {
            std::string error_str = std::string("unable to run model, error :") + error.Message();
            throw std::runtime_error(error_str);
        }
        std::shared_ptr<nic::InferResult> results_ptr;
        results_ptr.reset(results);

        auto cudaError = cudaMemcpy((void*)output_data_, output_cuda_buffer_->getInternalBuffer(), output_cuda_buffer_->size(), cudaMemcpyDeviceToHost);
        if(cudaError != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy output failed");
        }
        return processResults(output_data_, imagesRGB, confidence, nms_threshold);
    }

    void GRPCClient::print_stats() {
        nic::InferStat infer_stat;
        client_->ClientInferStat(&infer_stat);
        std::cout << "======Client Statistics======" << std::endl;
        std::cout << "completed_request_count " << infer_stat.completed_request_count
                  << std::endl;
        std::cout << "cumulative_total_request_time_ns "
                  << infer_stat.cumulative_total_request_time_ns / 1e6 << std::endl;
        std::cout << "cumulative_send_time_ns " << infer_stat.cumulative_send_time_ns / 1e6
                  << std::endl;
        std::cout << "cumulative_receive_time_ns "
                  << infer_stat.cumulative_receive_time_ns / 1e6 << std::endl;

        inference::ModelStatisticsResponse model_stat;
        client_->ModelInferenceStatistics(&model_stat, options_.model_name_);
        //std::cout << "======Model Statistics======" << std::endl;
        //std::cout << model_stat.DebugString() << std::endl;
    }


    std::vector<cv::Mat> GRPCClient::preprocessImage(const std::vector<cv::Mat> &imgs) {
        std::vector<cv::Mat> outputs;
        for(const auto &img: imgs) {
            int w, h, x, y;
            float r_w = INPUT_W / (img.cols*1.0f);
            float r_h = INPUT_H / (img.rows*1.0f);
            if (r_h > r_w) {
                w = INPUT_W;
                h = static_cast<int>(r_w * img.rows);
                x = 0;
                y = (INPUT_H - h) / 2;
            } else {
                w = static_cast<int>(r_h * img.cols);
                h = INPUT_H;
                x = (INPUT_W - w) / 2;
                y = 0;
            }
            cv::Mat re(h, w, CV_8UC3);
            cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
            cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
            re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
            outputs.push_back(out);
        }
        return outputs;
    }

    void GRPCClient::prepareBuffer(const std::vector<cv::Mat> &preparedImgs) {
        for(std::size_t batch = 0; batch < preparedImgs.size(); ++batch) {
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                input_data_[batch * 3 * INPUT_H * INPUT_W + i] = preparedImgs[batch].at<cv::Vec3b>(i)[2] / 255.0f;
                input_data_[batch * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = preparedImgs[batch].at<cv::Vec3b>(i)[1] / 255.0f;
                input_data_[batch * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = preparedImgs[batch].at<cv::Vec3b>(i)[0] / 255.0f;
            }
        }
        //size_t INPUT_BYTES = INPUT_DATA_SIZE * sizeof(float);
//        auto error =  input_->AppendRaw(
//                reinterpret_cast<uint8_t*>(input_data_),
//                INPUT_BYTES);
        //cudaMemcpy(input_cuda_buffer_->getInternalBuffer(), input_data_,
        auto cudaError = cudaMemcpy(input_cuda_buffer_->getInternalBuffer(), (void*)input_data_, input_cuda_buffer_->size(), cudaMemcpyHostToDevice);
        if(cudaError != cudaSuccess) {
             throw std::runtime_error("prepareBuffer cudaMemcpy has failed");
        }
//        if(!error.IsOk()) {
//           throw std::runtime_error("unable to set data for 'data'");
//        }
    }

    std::vector<common::datatypes::DetectionResults> GRPCClient::processResults(float *results, const std::vector<cv::Mat> &originalImgs, const float conf, const float nms_thresh) {

        std::vector<common::datatypes::DetectionResults> ret_results;
        //std::size_t idx = 0;
        //for(const auto &origin : oginalImgs) {
        for(std::size_t idx = 0; idx < originalImgs.size(); ++idx) {
            std::vector<YoloV5::Detection> res;
            helpers_utils::nms(res, &results[idx * OUTPUT_SIZE], conf, nms_thresh, classes_ids_);

            cv::Mat origin = originalImgs[idx];
            float r_w = INPUT_W / (origin.cols * 1.0f);
            float r_h = INPUT_H / (origin.rows * 1.0f);
            const int rows = origin.rows;
            const int cols = origin.cols;

            common::datatypes::DetectionResults outputs;

            if (r_h > r_w) {
                for (const auto &det : res) {
                    auto l = det.bbox[0] - det.bbox[2] / 2.f;
                    auto r = det.bbox[0] + det.bbox[2] / 2.f;
                    auto t = det.bbox[1] - det.bbox[3] / 2.f - (INPUT_H - r_w * rows) / 2;
                    auto b = det.bbox[1] + det.bbox[3] / 2.f - (INPUT_H - r_w * rows) / 2;
                    l = l / r_w;
                    r = r / r_w;
                    t = t / r_w;
                    b = b / r_w;
                    outputs.push_back({ common::datatypes::DetectionBox(l, t, r - l, b - t), static_cast<int>(det.class_id) });
                }
            } else {
                for (const auto &det : res) {
                    auto l = det.bbox[0] - det.bbox[2] / 2.f - (INPUT_W - r_h * cols) / 2;
                    auto r = det.bbox[0] + det.bbox[2] / 2.f - (INPUT_W - r_h * cols) / 2;
                    auto t = det.bbox[1] - det.bbox[3] / 2.f;
                    auto b = det.bbox[1] + det.bbox[3] / 2.f;
                    l = l / r_h;
                    r = r / r_h;
                    t = t / r_h;
                    b = b / r_h;
                    outputs.push_back({ common::datatypes::DetectionBox(l, t, r - l, b - t), static_cast<int>(det.class_id) });
                }
            }
            ret_results.push_back(outputs);
        }
        return ret_results;
    }
}
