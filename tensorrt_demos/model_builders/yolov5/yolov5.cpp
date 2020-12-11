#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "common/logging.h"
#include "common.hpp"
#include <filesystem>

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
//#define BATCH_SIZE 1

//INFO: refactored
//#define NET m  // s m l x
//#define NETSTRUCT(str) createEngine_##str
//#define CREATENET(net) NETSTRUCT(net)
//#define STR1(x) #x
//#define STR2(x) STR1(x)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = YoloV5::INPUT_H;
static const int INPUT_W = YoloV5::INPUT_W;
//static const int OUTPUT_SIZE = YoloV5::MAX_OUTPUT_BBOX_COUNT * sizeof(YoloV5::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
static const int OUTPUT_SIZE = YoloV5::OUTPUT_SIZE;
const char* INPUT_BLOB_NAME = YoloV5::INPUT_BLOB_NAME;
const char* OUTPUT_BLOB_NAME = YoloV5::OUTPUT_BLOB_NAME;
static Logger gLogger;

REGISTER_TENSORRT_PLUGIN(YoloV5PluginCreator);

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string &file, bool use_float16) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    //std::map<std::string, Weights> weightMap = loadWeights("../yolov5s.wts");
    std::map<std::string, Weights> weightMap = loadWeights(file);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolov5 head
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{2, 2}, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(256);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    Weights deconvwts15{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{2, 2}, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(128);
	//weightMap["deconv15"] = deconvwts15;

    ITensor* inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    IConvolutionLayer* conv18 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv19 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
    ITensor* inputTensors20[] = {conv19->getOutput(0), conv14->getOutput(0)};
    auto cat20 = network->addConcatenation(inputTensors20, 2);
    auto bottleneck_csp21 = bottleneckCSP(network, weightMap, *cat20->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.20");
    IConvolutionLayer* conv22 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv23 = convBnLeaky(network, weightMap, *bottleneck_csp21->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor* inputTensors24[] = {conv23->getOutput(0), conv10->getOutput(0)};
    auto cat24 = network->addConcatenation(inputTensors24, 2);
    auto bottleneck_csp25 = bottleneckCSP(network, weightMap, *cat24->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.23");
    IConvolutionLayer* conv26 = network->addConvolutionNd(*bottleneck_csp25->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloV5Layer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yolov5_layer", pluginData);
    ITensor* inputTensors_yolo[] = {conv26->getOutput(0), conv22->getOutput(0), conv18->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    if (use_float16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine_m(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string &file, bool use_float16) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(file);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 48, 3, "model.0");
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), 96, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 96, 96, 2, true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), 192, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 192, 192, 6, true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), 384, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 384, 384, 6, true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), 768, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 768, 768, 5, 9, 13, "model.8");
    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.9");
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), 384, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 384 * 2 * 2));
    for (int i = 0; i < 384 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 384 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 384, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(384);
    weightMap["deconv11"] = deconvwts11;
    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 768, 384, 2, false, 1, 0.5, "model.13");

    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), 192, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 192 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 192, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(192);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 384, 192, 2, false, 1, 0.5, "model.17");

    //yolo layer 1
    IConvolutionLayer* conv18 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv19 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), 192, 3, 2, 1, "model.18");

    ITensor* inputTensors20[] = { conv19->getOutput(0), conv14->getOutput(0) };
    auto cat20 = network->addConcatenation(inputTensors20, 2);

    auto bottleneck_csp21 = bottleneckCSP(network, weightMap, *cat20->getOutput(0), 384, 384, 2, false, 1, 0.5, "model.20");

    //yolo layer 2
    IConvolutionLayer* conv22 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv23 = convBnLeaky(network, weightMap, *bottleneck_csp21->getOutput(0), 384, 3, 2, 1, "model.21");

    ITensor* inputTensors24[] = { conv23->getOutput(0), conv10->getOutput(0) };
    auto cat24 = network->addConcatenation(inputTensors24, 2);

    auto bottleneck_csp25 = bottleneckCSP(network, weightMap, *cat24->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.23");

    // yolo layer 3
    IConvolutionLayer* conv26 = network->addConvolutionNd(*bottleneck_csp25->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloV5Layer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yolov5_layer", pluginData);
    ITensor* inputTensors_yolo[] = { conv26->getOutput(0), conv22->getOutput(0), conv18->getOutput(0) };
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    if (use_float16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine_l(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string &file, bool use_float16) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(file);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 64, 3, "model.0");
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), 128, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), 256, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 256, 256, 9, true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), 512, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 512, 512, 9, true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), 1024, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1024, 1024, 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1024, 1024, 3, false, 1, 0.5, "model.9");
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), 512, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 512 * 2 * 2));
    for (int i = 0; i < 512 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 512 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 512, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(512);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1024, 512, 3, false, 1, 0.5, "model.13");
    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), 256, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 256 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(256);
    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 512, 256, 3, false, 1, 0.5, "model.17");

    //yolo layer 1
    IConvolutionLayer* conv18 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv19 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), 256, 3, 2, 1, "model.18");

    // yolo layer 2
    ITensor* inputTensors20[] = { conv19->getOutput(0), conv14->getOutput(0) };
    auto cat20 = network->addConcatenation(inputTensors20, 2);

    auto bottleneck_csp21 = bottleneckCSP(network, weightMap, *cat20->getOutput(0), 512, 512, 3, false, 1, 0.5, "model.20");

    //yolo layer 3
    IConvolutionLayer* conv22 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv23 = convBnLeaky(network, weightMap, *bottleneck_csp21->getOutput(0), 512, 3, 2, 1, "model.21");

    ITensor* inputTensors24[] = { conv23->getOutput(0), conv10->getOutput(0) };
    auto cat24 = network->addConcatenation(inputTensors24, 2);

    auto bottleneck_csp25 = bottleneckCSP(network, weightMap, *cat24->getOutput(0), 1024, 1024, 3, false, 1, 0.5, "model.23");

    IConvolutionLayer* conv26 = network->addConvolutionNd(*bottleneck_csp25->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloV5Layer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yolov5_layer", pluginData);
    ITensor* inputTensors_yolo[] = { conv26->getOutput(0), conv22->getOutput(0), conv18->getOutput(0) };
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    if (use_float16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine_x(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string &file, bool use_float16) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(file);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 80, 3, "model.0");
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), 160, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 160, 160, 4, true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), 320, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 320, 320, 12, true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), 640, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 640, 640, 12, true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), 1280, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1280, 1280, 5, 9, 13, "model.8");

    /* ------- yolov5 head ------- */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1280, 1280, 4, false, 1, 0.5, "model.9");
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), 640, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 640 * 2 * 2));
    for (int i = 0; i < 640 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 640 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 640, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(640);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1280, 640, 4, false, 1, 0.5, "model.13");
    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), 320, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 320 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 320, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(320);
    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 640, 320, 4, false, 1, 0.5, "model.17");

    // yolo layer 1
    IConvolutionLayer* conv18 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv19 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), 320, 3, 2, 1, "model.18");

    ITensor* inputTensors20[] = { conv19->getOutput(0), conv14->getOutput(0) };
    auto cat20 = network->addConcatenation(inputTensors20, 2);

    auto bottleneck_csp21 = bottleneckCSP(network, weightMap, *cat20->getOutput(0), 640, 640, 4, false, 1, 0.5, "model.20");

    // yolo layer 2
    IConvolutionLayer* conv22 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv23 = convBnLeaky(network, weightMap, *bottleneck_csp21->getOutput(0), 640, 3, 2, 1, "model.21");

    ITensor* inputTensors24[] = { conv23->getOutput(0), conv10->getOutput(0) };
    auto cat24 = network->addConcatenation(inputTensors24, 2);

    auto bottleneck_csp25 = bottleneckCSP(network, weightMap, *cat24->getOutput(0), 1280, 1280, 4, false, 1, 0.5, "model.23");

    // yolo layer 3
    IConvolutionLayer* conv26 = network->addConvolutionNd(*bottleneck_csp25->getOutput(0), 3 * (YoloV5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloV5Layer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yolov5_layer", pluginData);
    ITensor* inputTensors_yolo[] = { conv26->getOutput(0), conv22->getOutput(0), conv18->getOutput(0) };
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    if (use_float16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

bool APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, const std::string name, const std::string &file, bool use_f16 = true) {
    // Create builder

    using createFuncT = std::function< ICudaEngine*(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string &file, bool use_f16)>;
    const std::map<std::string, createFuncT> builders_map = {
        {"yolov5s", createEngine_s},
        {"yolov5m", createEngine_m},
        {"yolov5l", createEngine_l},
        {"yolov5x", createEngine_x}
    };
    std::map<std::string, createFuncT>::const_iterator it = builders_map.find(name);
    if (it == builders_map.cend()) {
        std::cout << "Wrong builder name" << std::endl;
        return false;
    }

    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    //ICudaEngine* engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);
    ICudaEngine* engine = it->second(maxBatchSize, builder, config, DataType::kFLOAT, file, use_f16);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    return true;
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    unsigned int BATCH_SIZE = 1;

    if (argc > 1 && std::string(argv[1]) == "-s") {
        bool use_float16 = true;
        std::string weights_path;
        if (argc > 3 && std::string(argv[2]) == "no-f16") {
            use_float16 = false;
            weights_path = std::string(argv[3]);

            if(argc > 5 && std::string(argv[4]) == "--batch_size") {
                BATCH_SIZE = std::stoul(argv[5]);
            }
        } else {
            weights_path = std::string(argv[2]);
            if(argc > 4 && std::string(argv[3]) == "--batch_size") {
                BATCH_SIZE = std::stoul(argv[4]);
            }
        }

        auto model_name = std::filesystem::path(weights_path).stem().string();
        //std::string model_name(argv[2]);
        const std::string engine_name = model_name + ".engine";
        IHostMemory* modelStream{nullptr};
        if (APIToModel(BATCH_SIZE, &modelStream, model_name, weights_path, use_float16)) {
            assert(modelStream != nullptr);
            std::ofstream p(engine_name, std::ios::binary);
            if (!p) {
                std::cerr << "could not open plan output file" << std::endl;
                return -1;
            }
            p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
            modelStream->destroy();
        }
        return 0;
    } else if (argc == 4 && std::string(argv[1]) == "-d") {
        std::string model_name(argv[2]);
        const std::string engine_name = model_name + ".engine";
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else if(argc > 1 && std::string(argv[1]) == "-i") {
        std::string model_name(argv[2]);
        const std::string engine_name = model_name + ".engine";
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        delete[] trtModelStream;

        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        if (inputIndex != 0 && outputIndex != 1) {
            std::cout << "Invalid bind indexes for model : " << model_name << std::endl;
        }
        auto inDims = engine->getBindingDimensions(inputIndex);
        auto outDims = engine->getBindingDimensions(outputIndex);
        
        std::cout << "Input : " << std::endl;
        std::cout << "\t name : " << engine->getBindingName(inputIndex) << std::endl;
        std::cout << "\t" << engine->getBindingFormatDesc(inputIndex) << std::endl;
        std::cout << "\t dims : ";
        for(int i = 0; i < inDims.nbDims; ++i) {
           std::cout << inDims.d[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Output : " << std::endl;
        std::cout << "\t name : " << engine->getBindingName(outputIndex) << std::endl;
        std::cout << "\t" << engine->getBindingFormatDesc(outputIndex) << std::endl;
        std::cout << "\t dims : ";
        for(int i = 0; i < outDims.nbDims; ++i) {
           std::cout << outDims.d[i] << " ";
        }
        std::cout << std::endl;
      
        engine->destroy();
        runtime->destroy();
        return 0;
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s \"model_name\" [yolov5s, yolov5m, yolov5l, yolov5x] // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d \"model_name\" [yolov5s, yolov5m, yolov5l, yolov5x] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[3], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    std::vector<float> data(BATCH_SIZE * 3 * INPUT_H * INPUT_W);
    //static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    //static float prob[BATCH_SIZE * OUTPUT_SIZE];
    std::vector<float> prob(BATCH_SIZE * OUTPUT_SIZE);

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    unsigned int fcount = 0;
    for (unsigned int f = 0; f < (unsigned int)file_names.size(); ++f) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (unsigned int)file_names.size()) continue;
        for (unsigned int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(std::string(argv[3]) + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) {
                continue;
            }
            cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0f;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0f;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0f;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, &data[0], &prob[0], BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<YoloV5::Detection>> batch_res(fcount);
        for (unsigned int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (unsigned int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            cv::Mat img = cv::imread(std::string(argv[3]) + "/" + file_names[f - fcount + 1 + b]);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
