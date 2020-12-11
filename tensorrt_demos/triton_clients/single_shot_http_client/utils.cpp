#include "utils.h"
#include "common/yolov5/yolov5_layer.h"
#include <exception>
#include <stdexcept>
#include <opencv2/imgproc/imgproc.hpp>


namespace yolov5_utils {
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
            //std::cout << it->second[0].class_id << " --- " << std::endl;
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

    Client::Client (const std::string &url, const std::string &model_name, const std::vector<int> &classes_ids,  bool verbose)
        : classes_ids_(classes_ids)
        , options_(model_name)
        , input_data_(nullptr)
        , output_data_(nullptr)
    {
        auto error = nic::InferenceServerHttpClient::Create(&client_, url, verbose);
        if(!error.IsOk()) {
            throw std::runtime_error("Failed to create client");
        }
        std::vector<int64_t> shape{1, 3, INPUT_H, INPUT_W };
        // Initialize the inputs with the data.
        nic::InferInput* input;
        error = nic::InferInput::Create(&input, "data", shape, "FP32");
        if(!error.IsOk()) {
            throw std::runtime_error("unable to get input data");
        }
        input_.reset(input);

        //Milliseconds, please see common.h in triton_client/include
        constexpr const int CLIENT_TIMEOUT = 100;
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

        INPUT_DATA_SIZE = 3 * INPUT_H * INPUT_W;
        OUTPUT_DATA_SIZE = OUTPUT_SIZE;
        input_data_ = new float[INPUT_DATA_SIZE];
        output_data_ = new float[OUTPUT_DATA_SIZE];
    }

    Client::~Client() {
        if(input_data_) {
            delete[] input_data_;
            input_data_ = nullptr;
        }
        if(output_data_) {
            delete[] output_data_;
            output_data_ = nullptr;
        }
    }

    common::datatypes::DetectionResults Client::inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold) {
        cv::Mat preparedImg = preprocessImage(imageRGB);
        prepareBuffer(preparedImg);

        std::vector<nic::InferInput*> inputs{ input_.get() };
        std::vector<const nic::InferRequestedOutput*> outputs{ output_.get() };

        nic::InferResult* results;
        auto error =  client_->Infer(&results, options_, inputs, outputs);
        if(!error.IsOk()) {
            throw std::runtime_error("unable to run model");
        }
        std::shared_ptr<nic::InferResult> results_ptr;
        results_ptr.reset(results);

        float* output_data;
        std::size_t output_bytes;
        error = results_ptr->RawData(
                "prob", (const uint8_t **)&output_data, &output_bytes);
        if(!error.IsOk()) {
            throw std::runtime_error("unable to get result data for 'prob'");
        }
        if(output_bytes != OUTPUT_DATA_SIZE * sizeof(float)) {
            throw std::runtime_error("recieved wrong amount of data");
        }
        return processResults(output_data, imageRGB, confidence, nms_threshold);
    }

    cv::Mat Client::preprocessImage(const cv::Mat &img) {
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
        return out;
    }

    void Client::prepareBuffer(const cv::Mat &prepared) {
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            input_data_[i] = prepared.at<cv::Vec3b>(i)[2] / 255.0f;
            input_data_[i + INPUT_H * INPUT_W] = prepared.at<cv::Vec3b>(i)[1] / 255.0f;
            input_data_[i + 2 * INPUT_H * INPUT_W] = prepared.at<cv::Vec3b>(i)[0] / 255.0f;
        }
        size_t INPUT_BYTES = INPUT_DATA_SIZE * sizeof(float);
        auto error =  input_->AppendRaw(
                reinterpret_cast<uint8_t*>(input_data_),
                INPUT_BYTES);
        if(!error.IsOk()) {
           throw std::runtime_error("unable to set data for 'data'");
        }
    }

    common::datatypes::DetectionResults Client::processResults(float *results, const cv::Mat &origin, const float conf, const float nms_thresh) {
        std::vector<YoloV5::Detection> res;
        yolov5_utils::nms(res, results, conf, nms_thresh, classes_ids_);

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
        return outputs;
    }
}
