#include "stream_data_holder.h"

namespace ganz_camera {

    StreamDataHolder::StreamDataHolder()
        :stop_flag_(true)
    {}

    void StreamDataHolder::start(std::function<void(StreamDataHolder &holder, cv::Mat elem)> func) {
        stop_flag_ = false;
        while (!stop_flag_) {
            cv::Mat elem;
            if (data_.try_pop(elem)) {
                func(*this, elem);
            }
        }
    }

    void StreamDataHolder::stop() {
        stop_flag_ = true;
    }

    void StreamDataHolder::put(cv::Mat frame) {
        data_.push(frame);
    }
}