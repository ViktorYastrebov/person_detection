#include "deep_sort.h"
#include <fstream>

namespace deep_sort_tracker {

    using namespace common;

    DeepSort::DeepSort(const std::filesystem::path &model_path, const int BATCH_SIZE)
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
        const int inputIndex = engine.getBindingIndex("input.1");
        const int outputIndex = engine.getBindingIndex("208");
        if (inputIndex != 0 && outputIndex != 1) {
            throw std::runtime_error("Input/Output buffers ids are different");
        }

        cudaError_t error = cudaHostAlloc((void**)&input_host_buffer_, batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocPortable);
        error = cudaHostAlloc((void**)&output_host_buffer_, batch_size_ * OUTPUT_SIZE * sizeof(float), cudaHostAllocPortable);
        //input_host_buffer_ = new float[batch_size_ * 3 * INPUT_H * INPUT_W];
        //output_host_buffer_ = new float[batch_size_ * 3 * INPUT_H * INPUT_W];

        device_buffers_ = std::make_unique<DeviceBuffers>(batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float), batch_size_ * OUTPUT_SIZE * sizeof(float));
        error = cudaStreamCreate(&stream_);
        if (error != cudaSuccess) {
            throw std::runtime_error("Can't create CUDA stream");
        }
    }

    DeepSort::~DeepSort()
    {
        cudaFree(input_host_buffer_);
        cudaFree(output_host_buffer_);
    }

    cv::Mat DeepSort::getFeatures(const cv::Mat &imageRGB, const std::vector<cv::Rect> &bboxes) {
        cv::Mat total;
        int bbox_size = static_cast<int>(bboxes.size());

        int iters = bbox_size / batch_size_;
        for (int i = 0; i < iters; ++i) {
            std::vector<cv::Mat> prepared;
            for (int j = i * batch_size_; j < i * batch_size_ + batch_size_; ++j) {
                cv::Mat roi = imageRGB(bboxes[i]);
                cv::resize(roi, roi, cv::Size(INPUT_W, INPUT_H));
                prepared.push_back(roi);
            }
            preapreBuffer(prepared);

            cudaError_t error = cudaMemcpyAsync(device_buffers_->getBuffer(DeviceBuffers::INPUT), input_host_buffer_, batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream_);
            context_->enqueue(batch_size_, device_buffers_->getAll(), stream_, nullptr);
            error = cudaMemcpyAsync(output_host_buffer_, device_buffers_->getBuffer(DeviceBuffers::OUT), batch_size_ * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream_);
            error = cudaStreamSynchronize(stream_);
            cv::Mat result(cv::Size(OUTPUT_SIZE, batch_size_), CV_32FC1, output_host_buffer_);
            total.push_back(result);
        }

        int rest = bbox_size % batch_size_;
        std::vector<cv::Mat> prepared;
        for (int i = iters * batch_size_; i < iters * batch_size_ + rest; ++i) {
            cv::Mat roi = imageRGB(bboxes[i]);
            cv::resize(roi, roi, cv::Size(INPUT_W, INPUT_H));
            prepared.push_back(roi);
        }
        preapreBuffer(prepared);
        cudaError_t error = cudaMemcpyAsync(device_buffers_->getBuffer(DeviceBuffers::INPUT), input_host_buffer_, batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream_);
        context_->enqueue(batch_size_, device_buffers_->getAll(), stream_, nullptr);
        error = cudaMemcpyAsync(output_host_buffer_, device_buffers_->getBuffer(DeviceBuffers::OUT), batch_size_ * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        error = cudaStreamSynchronize(stream_);
        //TODO: USE Eigen here!

        //cv::Mat result(cv::Size(OUTPUT_SIZE, bbox_size), CV_32FC1, output_host_buffer_);
        //total.push_back(result);
        return total;
    }

    void DeepSort::preapreBuffer(const std::vector<cv::Mat> &crops) {
        int offset = 0;
        for (const auto &crop : crops) {
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                input_host_buffer_[offset + i] = crop.at<cv::Vec3b>(i)[2] / 255.0f;
                input_host_buffer_[offset + i + INPUT_H * INPUT_W] = crop.at<cv::Vec3b>(i)[1] / 255.0f;
                input_host_buffer_[offset + i + 2 * INPUT_H * INPUT_W] = crop.at<cv::Vec3b>(i)[0] / 255.0f;
            }
            offset += INPUT_H * INPUT_W;
        }
    }
}