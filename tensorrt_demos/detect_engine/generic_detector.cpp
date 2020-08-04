#include "generic_detector.h"
#include "common/buffer_manager.h"
#include <exception>
#include <iostream>

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

    GenericDetector::GenericDetector(const std::string &path, const int BATCH_SIZE)
        :model_path_(path)
        , BATCH_SIZE(BATCH_SIZE)
        , gLogger_(Logger::Severity::kVERBOSE)
    {}

    GenericDetector::~GenericDetector() = default;


    bool GenericDetector::buildEngine() {
        builder_ = TensorRTuniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger_.getTRTLogger()));
        if (!builder_)
        {
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network_ = TensorRTuniquePtr<nvinfer1::INetworkDefinition>(builder_->createNetworkV2(explicitBatch));
        if (!network_)
        {
            return false;
        }

        config_ = TensorRTuniquePtr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
        if (!config_)
        {
            return false;
        }

        auto parser = TensorRTuniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_, gLogger_.getTRTLogger()));
        if (!parser)
        {
            return false;
        }

        // https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/DataType.html
        // INFO: current model has uint8 as the input layer which is not supported by TensorRT
        auto parsed = parser->parseFromFile(model_path_.c_str(), static_cast<int>(gLogger_.getReportableSeverity()));
        if (!parsed)
        {
            return false;
        }
        builder_->setMaxBatchSize(BATCH_SIZE);

        //INFO: need to check how to check for DLA and DLA cores
        // enableDLA(builder.get(), config.get(), 10, true);

        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(builder_->buildEngineWithConfig(*network_, *config_), common::InferDeleter());
        if (!engine_)
        {
            return false;
        }
        input_dims_ = network_->getInput(0)->getDimensions();
        output_dims_ = network_->getOutput(0)->getDimensions();

        int n_outputs = network_->getNbOutputs();
        for (int i = 0; i < n_outputs; ++i) {
            std::cout << "Output " << i << ":" << std::endl;
            nvinfer1::Dims dims = network_->getOutput(i)->getDimensions();
            for (int dim_i = 0; dim_i < dims.nbDims; ++dim_i) {
                std::cout << "\t dim " << dim_i << " : " << dims.d[dim_i] << std::endl;
            }
        }

        return true;
    }

    bool GenericDetector::prepareBuffers() {
        buffers_ = std::make_unique<common::BufferManager>(engine_, BATCH_SIZE);
        context_ = TensorRTuniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        return context_ != nullptr;
    }

    GenericDetector::InputInfo GenericDetector::preprocessImage(const cv::Mat &imageRGB) {
        //TODO: generalize from input_dims_;
        cv::Size2f targetShape(640.0f, 640.0f);
        cv::Size2f actualSize(imageRGB.size().width, imageRGB.size().height);
        float ratio = std::min(targetShape.width / actualSize.width, targetShape.height / actualSize.height);
        cv::Size resizeShape(
            static_cast<int>(std::round(actualSize.width * ratio)),
            static_cast<int>(std::round(actualSize.height * ratio))
        );
        cv::Mat resized;
        cv::resize(imageRGB, resized, resizeShape);
        
        auto dw = targetShape.width - resizeShape.width;
        auto dh = targetShape.height - resizeShape.height;
        int top = static_cast<int>(std::round(dh / 2));
        int bottom = static_cast<int>(std::round(dh / 2));
        int left = static_cast<int>(std::round(dw / 2));
        int right = static_cast<int>(std::round(dw / 2));
        cv::Mat result;
        cv::copyMakeBorder(resized, result, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return InputInfo{ result, ratio, dw, dh };
    }

    cv::Mat GenericDetector::inference(const cv::Mat &imageRGB) {
        //YoloV3-SPP info:
        //  name: images
        //  type : float32[1, 3, 640, 640]

        std::string input_name(engine_->getBindingName(0));
        float *buffer = static_cast<float*>(buffers_->getHostBuffer(input_name));

        ///////////////////////////////////////////////
        auto info = preprocessImage(imageRGB);
        cv::Mat converted;
        cv::cvtColor(info.input, converted, cv::COLOR_BGR2RGB);

        cv::Mat fpMat;
        converted.convertTo(fpMat, CV_32FC3);
        fpMat /= 255.0f;

        std::vector<cv::Mat> channels;
        cv::split(fpMat, channels);

        for (int i = 0; i < channels.size(); ++i) {
            if (channels[i].isContinuous()) {

                //cv::Mat firstRow = channels[i].row(0);
                //std::cout << firstRow.colRange(0,3) << "..." << firstRow.colRange(firstRow.cols -3, firstRow.cols) << std::endl;

                auto total = channels[i].total();
                auto elem_size = channels[i].elemSize();
                size_t sizeInBytes = total * elem_size;
                std::memcpy(&buffer[total *i], channels[i].data, sizeInBytes);
            }
        }
        ///////////////////////////////////////////////
        buffers_->copyInputToDevice();

        bool status = context_->executeV2(buffers_->getDeviceBindings().data());
        if (!status) {
            return cv::Mat();
        }
        buffers_->copyOutputToHost();
        ///////////////////////////////////////////////

        cv::Mat result;

        //INFO: should be (3x80x80 + 3x40x40 + 3x20x20)
        int n_outputs = network_->getNbOutputs();
        for (int i = 0; i < n_outputs; ++i) {
            auto outputTensor = network_->getOutput(i);
            std::string out_name(outputTensor->getName());
            float* output = static_cast<float*>(buffers_->getHostBuffer(out_name));
            nvinfer1::Dims out_dims = outputTensor->getDimensions();
            cv::Mat reshaped_out(out_dims.d[0] * out_dims.d[1] * out_dims.d[2] * out_dims.d[3], out_dims.d[4], CV_32FC1, output);
            result.push_back(reshaped_out);
        }
        ///////////////////////////////////////////////
        // Now need to filter detection and apply anchors as far as I can see
        ///////////////////////////////////////////////
        for (int i = 0; i < result.rows; ++i) {
            float *ptr = result.ptr<float>(i);

        }
        return result;
    }
}