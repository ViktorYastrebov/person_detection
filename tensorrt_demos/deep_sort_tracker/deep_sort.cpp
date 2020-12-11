#include "deep_sort.h"
#include <fstream>


#include "NvInfer.h"
#include "NvInferRuntime.h"

#include <cuda_runtime_api.h>
#include "common/logging.h"
#include "common/common.h"
#include "deep_sort_types.h"

namespace deep_sort {

    using namespace common;

    struct DeepSort::Pimpl {
        Pimpl(const std::filesystem::path &model_path, const int BATCH_SIZE);
        ~Pimpl() = default;
        common::datatypes::Detections getFeatures(const cv::Mat &imageRGB, const common::datatypes::DetectionResults &bboxes);
    private:
        void preapreBuffer(const std::vector<cv::Mat> &resized);
    private:
        Logger gLogger_;
        int batch_size_;
        std::vector<char> deserialized_buffer_;
        common::TensorRTUPtr<nvinfer1::IRuntime> runtime_;
        common::TensorRTUPtr<nvinfer1::ICudaEngine> engine_;
        common::TensorRTUPtr<nvinfer1::IExecutionContext> context_;
        std::unique_ptr<common::HostBuffers> host_buffers_;
        std::unique_ptr<common::DeviceBuffers> device_buffers_;
        cudaStream_t stream_;
    };

    DeepSort::Pimpl::Pimpl(const std::filesystem::path &model_path, const int BATCH_SIZE)
        :batch_size_(BATCH_SIZE)
    {
        if (batch_size_ > MAX_BATCH_SIZE) {
            throw std::logic_error("MAX SIZE overhead, regenerate model with the different max batch size");
        }

        std::ifstream ifs(model_path.string().c_str(), std::ios_base::binary);
        if (ifs) {
            deserialized_buffer_ = std::vector<char>(
                (std::istreambuf_iterator<char>(ifs)),
                std::istreambuf_iterator<char>()
                );
        }
        else {
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
        const int inputIndex = engine.getBindingIndex("input.1");
        const int outputIndex = engine.getBindingIndex("208");
        if (inputIndex != 0 && outputIndex != 1) {
            throw std::runtime_error("Input/Output buffers ids are different");
        }
        const int INPUT_BUFFER_SIZE = batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float);
        const int OUTPUT_BUFFER_SIZE = batch_size_ * OUTPUT_SIZE * sizeof(float);
        host_buffers_ = std::make_unique<common::HostBuffers>(INPUT_BUFFER_SIZE, OUTPUT_BUFFER_SIZE);
        device_buffers_ = std::make_unique<DeviceBuffers>(INPUT_BUFFER_SIZE, OUTPUT_BUFFER_SIZE);
        cudaError_t error = cudaStreamCreate(&stream_);
        if (error != cudaSuccess) {
            throw std::runtime_error("Can't create CUDA stream");
        }
    }

    common::datatypes::Detections DeepSort::Pimpl::getFeatures(const cv::Mat &imageRGB, const common::datatypes::DetectionResults &bboxes) {
        int bbox_size = static_cast<int>(bboxes.size());

        common::datatypes::Detections result;

        auto frame_size = imageRGB.size();

        int iters = bbox_size / batch_size_;
        for (int i = 0; i < iters; ++i) {
            std::vector<cv::Mat> prepared;
            for (int j = i * batch_size_; j < i * batch_size_ + batch_size_; ++j) {
                cv::Rect rect_roi(static_cast<int>(std::roundf(bboxes[j].bbox(0))),
                                  static_cast<int>(std::roundf(bboxes[j].bbox(1))),
                                  static_cast<int>(std::roundf(bboxes[j].bbox(2))),
                                  static_cast<int>(std::roundf(bboxes[j].bbox(3)))
                );
                rect_roi.x = std::max(rect_roi.x, 0);
                rect_roi.y = std::max(rect_roi.y, 0);

                if ((rect_roi.x + rect_roi.width) >= frame_size.width) {
                    rect_roi.width = frame_size.width - 1 - rect_roi.x;
                }
                if ((rect_roi.y + rect_roi.height) >= frame_size.height) {
                    rect_roi.height = frame_size.height - 1 - rect_roi.y;
                }
                cv::Mat roi = imageRGB(rect_roi);
                cv::resize(roi, roi, cv::Size(INPUT_W, INPUT_H));
                prepared.push_back(roi);
            }
            preapreBuffer(prepared);

            cudaError_t error = cudaMemcpyAsync(device_buffers_->getBuffer(BUFFER_TYPE::INPUT), host_buffers_->getBuffer(BUFFER_TYPE::INPUT), batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream_);
            context_->enqueue(batch_size_, device_buffers_->getAll(), stream_, nullptr);
            error = cudaMemcpyAsync(host_buffers_->getBuffer(BUFFER_TYPE::OUT), device_buffers_->getBuffer(BUFFER_TYPE::OUT), batch_size_ * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream_);
            error = cudaStreamSynchronize(stream_);

            float *ptr = static_cast<float*>(host_buffers_->getBuffer(BUFFER_TYPE::OUT));
            for (int j = 0; j < batch_size_; ++j) {
                common::datatypes::Feature feature(&ptr[j*OUTPUT_SIZE]);
                result.push_back({ bboxes[i*batch_size_ + j].bbox,  feature, bboxes[i*batch_size_ + j].class_id });
            }
        }
        int rest = bbox_size % batch_size_;
        std::vector<cv::Mat> prepared;
        for (int i = iters * batch_size_; i < iters * batch_size_ + rest; ++i) {
            //INFO: check could work as is
            cv::Rect rect_roi(
                static_cast<int>(std::roundf(bboxes[i].bbox(0))),
                static_cast<int>(std::roundf(bboxes[i].bbox(1))),
                static_cast<int>(std::roundf(bboxes[i].bbox(2))),
                static_cast<int>(std::roundf(bboxes[i].bbox(3)))
            );
            rect_roi.x = std::max(rect_roi.x, 0);
            rect_roi.y = std::max(rect_roi.y, 0);

            if ((rect_roi.x + rect_roi.width) >= frame_size.width) {
                rect_roi.width = frame_size.width - 1 - rect_roi.x;
            }
            if ((rect_roi.y + rect_roi.height) >= frame_size.height) {
                rect_roi.height = frame_size.height - 1 - rect_roi.y;
            }
            cv::Mat roi = imageRGB(rect_roi);
            cv::resize(roi, roi, cv::Size(INPUT_W, INPUT_H));
            prepared.push_back(roi);
        }
        preapreBuffer(prepared);
        cudaError_t error = cudaMemcpyAsync(device_buffers_->getBuffer(BUFFER_TYPE::INPUT), host_buffers_->getBuffer(BUFFER_TYPE::INPUT), batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream_);
        context_->enqueue(batch_size_, device_buffers_->getAll(), stream_, nullptr);
        error = cudaMemcpyAsync(host_buffers_->getBuffer(BUFFER_TYPE::OUT), device_buffers_->getBuffer(BUFFER_TYPE::OUT), batch_size_ * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        error = cudaStreamSynchronize(stream_);
        //TODO: USE Eigen here!

        float *ptr = static_cast<float*>(host_buffers_->getBuffer(BUFFER_TYPE::OUT));
        for (int i = 0; i < rest; ++i) {
            common::datatypes::Feature feature(&ptr[i*OUTPUT_SIZE]);
            result.push_back({ bboxes[iters*batch_size_ + i].bbox,  feature, bboxes[iters*batch_size_ + i].class_id });
        }
        return result;
    }

    void DeepSort::Pimpl::preapreBuffer(const std::vector<cv::Mat> &crops) {
        int offset = 0;
        float * input_host_buffer = (float*)host_buffers_->getBuffer(BUFFER_TYPE::INPUT);
        for (const auto &crop : crops) {
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                input_host_buffer[offset + i] = crop.at<cv::Vec3b>(i)[2] / 255.0f;
                input_host_buffer[offset + i + INPUT_H * INPUT_W] = crop.at<cv::Vec3b>(i)[1] / 255.0f;
                input_host_buffer[offset + i + 2 * INPUT_H * INPUT_W] = crop.at<cv::Vec3b>(i)[0] / 255.0f;
            }
            offset += INPUT_H * INPUT_W;
        }
    }

    DeepSort::DeepSort(const std::filesystem::path &model_path, const int BATCH_SIZE)
        : pimpl_(nullptr)
    {
        try {
            pimpl_ = new Pimpl(model_path, BATCH_SIZE);
        }
        catch (...) {
            delete pimpl_;
            pimpl_ = nullptr;
            throw;
        }
    }

    DeepSort::~DeepSort() {
        if (pimpl_) {
            delete pimpl_;
        }
    }

    common::datatypes::Detections DeepSort::getFeatures(const cv::Mat &imageRGB, const common::datatypes::DetectionResults &bboxes) {
        return pimpl_->getFeatures(imageRGB, bboxes);
        
    }
}