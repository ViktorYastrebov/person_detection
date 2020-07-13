#pragma once

#include <opencv2/dnn/dnn.hpp>
#include <vector>

#include "base_model.h"
#include "device_utils.h"


#pragma warning(push)
#pragma warning(disable: 4251)

//INFO: really low acuuracy + known problem with the rescaling the detected BBOX-es
//       Implemented only for testing
class ENGINE_DECL YoloV3Tiny : public BaseModel {
public:
    YoloV3Tiny(const std::string &model, const std::string &config, const std::vector<int> &classes, const float confidence = 0.3, RUN_ON device = RUN_ON::CPU);

    virtual ~YoloV3Tiny() = default;
    virtual std::vector<DetectionResult> process(const cv::Mat &frame);
protected:
    const int INPUT_SIZE = 224;
    const int BATCH_SIZE = 128;
private:
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
    const float conf_threshold_;
    std::vector<int> filtered_classes_;
};

#pragma warning(pop)