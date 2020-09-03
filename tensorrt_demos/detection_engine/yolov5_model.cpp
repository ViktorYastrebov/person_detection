#include "yolov5_model.h"
#include "common/yolov5/yolov5_layer.h"
#include "cuda_runtime_api.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <map>

namespace {
    using namespace nvinfer1;
    REGISTER_TENSORRT_PLUGIN(YoloV5PluginCreator);
}

namespace detector {

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


    using namespace common;
    using namespace YoloV5;

    YoloV5Model::YoloV5Model(const std::filesystem::path &model_path, const std::vector<int> &classes_ids, const int BATCH_SIZE)
        :CommonDetector()
        , classes_ids_(classes_ids)
    {
        batch_size_ = BATCH_SIZE;

        std::ifstream ifs(model_path.string().c_str(), std::ios_base::binary);
        if (ifs) {
            deserialized_buffer_ = std::vector<char>(
                (std::istreambuf_iterator<char>(ifs)),
                std::istreambuf_iterator<char>()
                );
        } else {
            throw std::runtime_error("Model path does not exists");
        }
        runtime_ = TensorRTUPtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger_));
        if (!runtime_) {
            throw std::runtime_error("Can't create IRuntime");
        }
        engine_ = TensorRTUPtr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(deserialized_buffer_.data(), deserialized_buffer_.size()));
        if (!engine_) {
            throw std::runtime_error("Can't create ICudaEngine");
        }
        context_ = TensorRTUPtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

        const nvinfer1::ICudaEngine& engine = context_->getEngine();
        if (engine.getNbBindings() != 2) {
            throw std::runtime_error("Buffers amount does not equal 2, might different model is used");
        }
        const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
        if (inputIndex != 0 && outputIndex != 1) {
            throw std::runtime_error("Input/Output buffers ids are different");
        }
        const int INPUT_BUFFER_SIZE = batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float);
        const int OUTPUT_BUFFER_SIZE = batch_size_ * OUTPUT_SIZE * sizeof(float);

        host_buffers_ = std::make_unique<HostBuffers>(INPUT_BUFFER_SIZE, OUTPUT_BUFFER_SIZE);
        device_buffers_ = std::make_unique<DeviceBuffers>(INPUT_BUFFER_SIZE, OUTPUT_BUFFER_SIZE);
        cudaError_t error = cudaStreamCreate(&stream_);
        if (error != cudaSuccess) {
            throw std::runtime_error("Can't create CUDA stream");
        }
    }

    common::datatypes::DetectionResults YoloV5Model::inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold) {
        cv::Mat prepared = preprocessImage(imageRGB);
        prepareBuffer(prepared);
        cudaError_t error = cudaMemcpyAsync(device_buffers_->getBuffer(BUFFER_TYPE::INPUT), host_buffers_->getBuffer(BUFFER_TYPE::INPUT), batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream_);
        context_->enqueue(batch_size_, device_buffers_->getAll(), stream_, nullptr);
        error = cudaMemcpyAsync(host_buffers_->getBuffer(BUFFER_TYPE::OUT), device_buffers_->getBuffer(BUFFER_TYPE::OUT), batch_size_ * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        error = cudaStreamSynchronize(stream_);
        return processResults(imageRGB, confidence, nms_threshold);
    }

    cv::Mat YoloV5Model::preprocessImage(const cv::Mat &img) {
        int w, h, x, y;
        float r_w = INPUT_W / (img.cols*1.0f);
        float r_h = INPUT_H / (img.rows*1.0f);
        if (r_h > r_w) {
            w = INPUT_W;
            h = r_w * img.rows;
            x = 0;
            y = (INPUT_H - h) / 2;
        } else {
            w = r_h * img.cols;
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

    void YoloV5Model::prepareBuffer(cv::Mat &prepared) {
        float * input_host_buffer_ = (float*)host_buffers_->getBuffer(BUFFER_TYPE::INPUT);
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            input_host_buffer_[i] = prepared.at<cv::Vec3b>(i)[2] / 255.0f;
            input_host_buffer_[i + INPUT_H * INPUT_W] = prepared.at<cv::Vec3b>(i)[1] / 255.0f;
            input_host_buffer_[i + 2 * INPUT_H * INPUT_W] = prepared.at<cv::Vec3b>(i)[0] / 255.0f;
        }
    }

    common::datatypes::DetectionResults YoloV5Model::processResults(const cv::Mat &prepared, const float conf, const float nms_thresh) {
        std::vector<YoloV5::Detection> res;
        float *output_host_buffer = (float*)host_buffers_->getBuffer(BUFFER_TYPE::OUT);
        yolov5_utils::nms(res, output_host_buffer, conf, nms_thresh, classes_ids_);

        float r_w = INPUT_W / (prepared.cols * 1.0f);
        float r_h = INPUT_H / (prepared.rows * 1.0f);
        const int rows = prepared.rows;
        const int cols = prepared.cols;

        common::datatypes::DetectionResults outputs;

        if (r_h > r_w) {
            for (const auto &det : res) {
                int l = det.bbox[0] - det.bbox[2] / 2.f;
                int r = det.bbox[0] + det.bbox[2] / 2.f;
                int t = det.bbox[1] - det.bbox[3] / 2.f - (INPUT_H - r_w * rows) / 2;
                int b = det.bbox[1] + det.bbox[3] / 2.f - (INPUT_H - r_w * rows) / 2;
                l = l / r_w;
                r = r / r_w;
                t = t / r_w;
                b = b / r_w;
                outputs.push_back({ common::datatypes::DetectionBox(l, t, r - l, b - t), static_cast<int>(det.class_id) });
            }
        } else {
            for (const auto &det : res) {
                int l = det.bbox[0] - det.bbox[2] / 2.f - (INPUT_W - r_h * cols) / 2;
                int r = det.bbox[0] + det.bbox[2] / 2.f - (INPUT_W - r_h * cols) / 2;
                int t = det.bbox[1] - det.bbox[3] / 2.f;
                int b = det.bbox[1] + det.bbox[3] / 2.f;
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
