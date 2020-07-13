#include <opencv2/dnn/dnn.hpp>
#include <vector>

#include "base_model.h"
#include "device_utils.h"


#pragma warning(push)
#pragma warning(disable: 4251)

class ENGINE_DECL YoloV3SPP : public BaseModel {
public:
    YoloV3SPP(const std::string &model, const std::vector<int> &classes, const float confidence = 0.3, RUN_ON device = RUN_ON::CPU);

    virtual ~YoloV3SPP() = default;
    virtual std::vector<DetectionResult> process(const cv::Mat &frame);
private:
    cv::dnn::Net net_;
    std::vector<cv::String> output_layers_;
    const float conf_threshold_;
    std::vector<int> filtered_classes_;
};

#pragma warning(pop)