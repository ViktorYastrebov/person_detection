#include "yolov3_batch.h"

#include <iostream>

YoloV3Batched::YoloV3Batched(const std::string &path, const std::string &config, const std::vector<int> &classes, const float confidence, RUN_ON device)
    : conf_threshold_(confidence)
    , filtered_classes_(classes)
{
    net_ = cv::dnn::readNet(path, config);
    if (device == RUN_ON::GPU) {
        net_.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
    }
    output_layers_ = net_.getUnconnectedOutLayersNames();
}


//INFO: in future is OpenCV DNN module will support to pass GpuMat & return GPU mat 
//      it's possible to optmize also the data extraction by:
//      parallel search by rows the max values

std::vector<std::vector<DetectionResult>> YoloV3Batched::process(const std::vector<cv::Mat> &frames) {
    constexpr const double NORM_FACTOR = 1.0 / 255.0;
    constexpr const int PERSON_CLASS_ID = 0;

    auto blob = cv::dnn::blobFromImages(frames, NORM_FACTOR, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(0, 0, 0), true, false);
    net_.setInput(blob);

    std::vector<std::vector<cv::Mat> > ret;
    net_.forward(ret, output_layers_);

    auto start = std::chrono::system_clock::now();

    std::vector<std::vector<DetectionResult>> output(frames.size());
    std::vector<std::vector<cv::Rect >> bboxes(frames.size());
    std::vector < std::vector<float>> scores(frames.size());
    std::vector < std::vector<int>> classes(frames.size());

    for (const auto &l1 : ret) {
        for (const auto &mat : l1) {
            std::vector<cv::Mat> slides(frames.size());
            int planes = mat.size[0];
            int rows = mat.size[1];
            int cols = mat.size[2];
            cv::Mat m2(rows, cols*planes, CV_32FC1, mat.data);
            cv::Mat mat2planes = m2.reshape(frames.size());
            cv::split(mat2planes, slides);
            int idx = 0;
            for (auto &slide : slides) {
                auto width = frames[idx].size().width;
                auto height = frames[idx].size().height;

                double min = 0;
                double max = 0;
                cv::Point minLoc;
                cv::Point maxLoc;

                for (int row_it = 0; row_it < slide.rows; ++row_it) {
                    const float * row_ptr = slide.ptr<float>(row_it);
#if 1
                    auto value_it = std::max_element(&row_ptr[5], &row_ptr[slide.cols]);
                    auto value = *value_it;
                    std::size_t class_id = std::distance(&row_ptr[5], value_it);
#else 
                    int class_id = -1;
                    cv::minMaxIdx(row, nullptr, &max, nullptr, &class_id);
                    float value = static_cast<float>(max);
#endif

                    auto it = std::find(filtered_classes_.cbegin(), filtered_classes_.cend(), class_id);
                    if (it != filtered_classes_.cend() && value > conf_threshold_) {
                        int center_x = static_cast<int>(row_ptr[0] * width);
                        int center_y = static_cast<int>(row_ptr[1] * height);
                        int w = static_cast<int>(row_ptr[2] * width);
                        int h = static_cast<int>(row_ptr[3] * height);
                        int x = static_cast<int>(center_x - w / 2);
                        int y = static_cast<int>(center_y - h / 2);
                        bboxes[idx].push_back(cv::Rect(x, y, w, h));
                        scores[idx].push_back(value);
                        classes[idx].push_back(class_id);
                    }
                }
                ++idx;
            }
        }
    }

    std::size_t amount = frames.size();
    for (std::size_t i = 0; i < amount; ++i) {
        std::vector<int> idxs;
        cv::dnn::NMSBoxes(bboxes[i], scores[i], 0.5f, 0.4f, idxs);
        for (const auto &idx : idxs) {
            output[i].push_back({ bboxes[i][idx], classes[i][idx] });
        }
    }

    auto end = std::chrono::system_clock::now();
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "getting data time: " << int_ms.count() << " ms" << std::endl;
    return output;
}