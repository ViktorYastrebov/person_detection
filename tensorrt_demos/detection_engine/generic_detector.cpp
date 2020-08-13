#include "generic_detector.h"
#include <exception>
#include <iostream>
#include <fstream>
#include "yololayer.h"

namespace {
    using namespace nvinfer1;
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
}


namespace helper {
    std::string type2str(int type) {
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
        }

        r += "C";
        r += (chans + '0');

        return r;
    }

    //void enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true) {
    //    if (useDLACore >= 0)
    //    {
    //        if (builder->getNbDLACores() == 0)
    //        {
    //            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
    //                << std::endl;
    //            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
    //        }
    //        if (allowGPUFallback)
    //        {
    //            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    //        }
    //        if (!builder->getInt8Mode() && !config->getFlag(nvinfer1::BuilderFlag::kINT8))
    //        {
    //            // User has not requested INT8 Mode.
    //            // By default run in FP16 mode. FP32 mode is not permitted.
    //            builder->setFp16Mode(true);
    //            config->setFlag(nvinfer1::BuilderFlag::kFP16);
    //        }
    //        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    //        config->setDLACore(useDLACore);
    //        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    //    }
    //}
}

namespace detection_engine {
    using namespace common;

    float GenericDetector::Utils::iou(float lbox[4], float rbox[4]) {
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

    bool GenericDetector::Utils::cmp(Detection& a, Detection& b) {
        return a.det_confidence > b.det_confidence;
    }

    void GenericDetector::Utils::nms(std::vector<Detection>& res, float *output, const float conf, const float nms_thresh) {
        std::map<float, std::vector<Detection>> m;
        for(int i = 0; i < output[0] && i < MAX_OUTPUT_COUNT; ++i) {
        //for (int i = 0; i < output[0] && i < 1000; i++) {
            if (output[1 + 7 * i + 4] <= conf) continue;
            Detection det;
            memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
            if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
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



    GenericDetector::GenericDetector(const std::filesystem::path &model_path, const int BATCH_SIZE)
        :gLogger_()
        , batch_size_(BATCH_SIZE)
    {
        std::ifstream ifs(model_path.string().c_str(), std::ios_base::binary);
        if (ifs) {
            deserialized_buffer_ =  std::vector<char>(
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

    std::vector<common::datatypes::DetectionBox> GenericDetector::inference(const cv::Mat &imageRGB, const float confidence, const float nms_threshold) {
        cv::Mat prepared = preprocessImage(imageRGB);
        preapreBuffer(prepared);
        cudaError_t error = cudaMemcpyAsync(device_buffers_->getBuffer(BUFFER_TYPE::INPUT), host_buffers_->getBuffer(BUFFER_TYPE::INPUT), batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream_);
        context_->enqueue(batch_size_, device_buffers_->getAll(), stream_, nullptr);
        error = cudaMemcpyAsync(host_buffers_->getBuffer(BUFFER_TYPE::OUT), device_buffers_->getBuffer(BUFFER_TYPE::OUT), batch_size_ * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        error = cudaStreamSynchronize(stream_);
        return processResults(imageRGB, confidence, nms_threshold);
    }

    cv::Mat GenericDetector::preprocessImage(const cv::Mat &img) {
        int w, h, x, y;
        float r_w = INPUT_W / (img.cols*1.0);
        float r_h = INPUT_H / (img.rows*1.0);
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

    void GenericDetector::preapreBuffer(cv::Mat &prepared) {
        //TODO: may be there is faster way
        float * input_host_buffer_ = (float*)host_buffers_->getBuffer(BUFFER_TYPE::INPUT);
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            input_host_buffer_[i] = prepared.at<cv::Vec3b>(i)[2] / 255.0f;
            input_host_buffer_[i + INPUT_H * INPUT_W] = prepared.at<cv::Vec3b>(i)[1] / 255.0f;
            input_host_buffer_[i + 2 * INPUT_H * INPUT_W] = prepared.at<cv::Vec3b>(i)[0] / 255.0f;
        }
    }

    std::vector<common::datatypes::DetectionBox> GenericDetector::processResults(const cv::Mat &prepared, const float conf, const float nms_thresh) {
        std::vector<Detection> res;
        float *output_host_buffer = (float*)host_buffers_->getBuffer(BUFFER_TYPE::OUT);
        Utils::nms(res, output_host_buffer, conf, nms_thresh);

        float r_w = INPUT_W / (prepared.cols * 1.0);
        float r_h = INPUT_H / (prepared.rows * 1.0);
        const int rows = prepared.rows;
        const int cols = prepared.cols;

        std::vector<common::datatypes::DetectionBox> outputs;

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
                outputs.push_back(common::datatypes::DetectionBox(l, t, r - l, b - t));
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
                outputs.push_back(common::datatypes::DetectionBox(l, t, r - l, b - t));
            }
        }
        return outputs;
    }
}