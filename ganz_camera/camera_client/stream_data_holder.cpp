#include "stream_data_holder.h"

namespace ganz_camera {

    StreamDataHolder::StreamDataHolder()
        :stop_flag_(true)
    {}

    void StreamDataHolder::start(EntryProcessFunc func) {
        stop_flag_ = false;
        while (!stop_flag_) {
            cv::Mat elem;
            if (data_.try_pop(elem)) {
                // INFO: return value of try_pop from faces does not metter,
                //       can be empty -> means no faces detected
                FaceDataVector faces;
                faces_.try_pop(faces);
                func(*this, elem, faces);
            }
        }
    }

    void StreamDataHolder::stop() {
        stop_flag_ = true;
    }

    void StreamDataHolder::put(cv::Mat frame) {
        data_.push(frame);
    }

    void StreamDataHolder::put(FaceDataVector &&faces) {
        faces_.push(faces);
    }

}