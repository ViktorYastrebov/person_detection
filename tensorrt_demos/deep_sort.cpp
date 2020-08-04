#include "deep_sort.h"
#include "buffer_manager.h"
//#include <fstream>
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
}

namespace tensorrt_inference {
    DeepSortModel::DeepSortModel(const std::string &path, const int BATCH_SIZE)
        :model_path_(path)
        , BATCH_SIZE(BATCH_SIZE)
        , gLogger_(Logger::Severity::kVERBOSE)
        , images_processed_(0)
    {}

    bool DeepSortModel::prepareModel() {
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
        //builder_->setMaxBatchSize(BATCH_SIZE);

        //INFO: need to check how to check for DLA and DLA cores
        // enableDLA(builder.get(), config.get(), 10, true);

        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(builder_->buildEngineWithConfig(*network_, *config_), InferDeleter());
        if (!engine_)
        {
            return false;
        }

        //assert(network->getNbInputs() == 1);
        input_dims_ = network_->getInput(0)->getDimensions();
        //assert(mInputDims.nbDims == 4);

        //assert(network->getNbOutputs() == 1);
        output_dims_ = network_->getOutput(0)->getDimensions();
        //assert(mOutputDims.nbDims == 2);
        return true;
    }

    void DeepSortModel::setInputPath(const std::filesystem::path &path) {
        input_path_ = path;
    }

    //INFO: ideas, the size of image is constant: 64x128x3 so pinned memory should not be even change by size,
    //      I mean reallocations
    bool DeepSortModel::processInput(const BufferManager& buffers) {

        //name: input.1
        //type : float32[32, 3, 128, 64]
        const std::string input_name = "input.1";

        float *buffer = static_cast<float*>(buffers.getHostBuffer(input_name));

        images_processed_ = 0;

        constexpr const std::size_t IMAG_SIZE = 64 * 128 * 3 * sizeof(float);

        for (auto& p : std::filesystem::recursive_directory_iterator(input_path_)) {
            if (p.path().extension() == ".png") {
                cv::Mat img = cv::imread(p.path().string());
                if (!img.empty()) {
                    //INFO: OK as far as I can see it's the linear memory so the dimentions could be like:
                    //     Batch(i) * image_size, which is 3x128x64

                    //std::cout << helper::type2str(img.type()) << std::endl;

                    std::vector<float> means{ 0.485f, 0.456f, 0.406f };
                    std::vector<float> stds{ 0.229f, 0.224f, 0.225f };
                    cv::Mat fpMat;
                    img.convertTo(fpMat, CV_32FC3);
                    //std::cout << helper::type2str(fpMat.type()) << std::endl;

                    fpMat = fpMat / 255.0f;
                    std::vector<cv::Mat> channels;
                    cv::split(fpMat, channels);
                    for (int i = 0; i < channels.size(); ++i) {
                        channels[i] = (channels[i] - means[i]) / stds[i];
                        //std::cout << helper::type2str(channels[i].type()) << std::endl;

                        //cv::Mat firstRow = channels[i].row(0);
                        //std::cout << firstRow.colRange(0,3) << "..." << firstRow.colRange(firstRow.cols -3, firstRow.cols) << std::endl;

                        if (channels[i].isContinuous()) {
                            size_t sizeInBytes = channels[i].total() * channels[i].elemSize();
                            std::memcpy(&buffer[IMAG_SIZE*images_processed_ + sizeInBytes*i], channels[i].data, sizeInBytes);
                        }
                    }
                    ++images_processed_;
                }
            }
        }
        return true;
    }


    bool DeepSortModel::inference() {
        BufferManager buffers(engine_, BATCH_SIZE);

        auto context = TensorRTuniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context)
        {
            return false;
        }

        // Read the input data into the managed buffers
        //assert(mParams.inputTensorNames.size() == 1);
        if (!processInput(buffers))
        {
            return false;
        }

        // Memcpy from host input buffers to device input buffers
        buffers.copyInputToDevice();

        bool status = context->executeV2(buffers.getDeviceBindings().data());
        if (!status)
        {
            return false;
        }

        // Memcpy from device output buffers to host output buffers
        buffers.copyOutputToHost();

        // Verify results
        if (!verifyOutput(buffers))
        {
            return false;
        }
        return true;
    }

    bool DeepSortModel::verifyOutput(const BufferManager& buffers) {
        //name: 208
        // type: float32[32, 512]
        // But it depends on the input size also(means if it's 22 images so the actual memory is: 22x512. The reset contains the rubish
        const std::string out_name = "208";
        //output_dims_.d
        //const int outputSize = mOutputDims.d[1];
        constexpr const std::size_t SIZE_STEP = 512;
        
        float* output = static_cast<float*>(buffers.getHostBuffer(out_name));
        cv::Mat result(cv::Size(SIZE_STEP, images_processed_), CV_32F, output);

        cv::Mat row1 = result.row(0);
        std::cout << row1.colRange(0,3) << "..." << row1.colRange(row1.cols -3, row1.cols) << std::endl;

        cv::Mat row2 = result.row(1);
        std::cout << row2.colRange(0, 3) << "..." << row2.colRange(row2.cols - 3, row2.cols) << std::endl;

        return true;
    }
}