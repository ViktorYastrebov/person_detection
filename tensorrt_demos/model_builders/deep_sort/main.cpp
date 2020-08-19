#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h" 
#include <cuda_runtime_api.h>

#include "common/common.h"
#include "common/logging.h"
#include <filesystem>
#include <fstream>

#include <iostream>


//INFO: need to check about the batching because exported ONNX model can be different
//      By default it's BATCH_SIZE = 32

bool buildEngine(const std::filesystem::path &model_path, const int MAX_BATCH_SIZE = 32) {
    using namespace common;
    Logger gLogger_;
    auto builder = TensorRTUPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger_.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    builder->setMaxBatchSize(MAX_BATCH_SIZE);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = TensorRTUPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = TensorRTUPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger_.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto parsed = parser->parseFromFile(model_path.string().c_str(), static_cast<int>(gLogger_.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), InferDeleter());
    if (!engine)
    {
        return false;
    }
    auto hostMemory = engine->serialize();
    const std::string out_name = model_path.stem().string() + ".engine";
    auto output = model_path.parent_path() / out_name;
    std::ofstream ofs(output.string(), std::ios::binary);
    if (!ofs) {
        std::cerr << "could not open plan output file" << std::endl;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(hostMemory->data()), hostMemory->size());
    return true;
}


int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cout << "wrong amount of arguments, usage: deep_sort_builder.exe \"model_path.onnx\"" << std::endl;
        return 0;
    }
    std::filesystem::path model_path(argv[1]);
    std::cout << "Generating model engine, might take time ... " << std::endl;
    if (buildEngine(model_path)) {
        std::cout << "Generate model engine successfuly" << std::endl;
    }

    return 0;
}