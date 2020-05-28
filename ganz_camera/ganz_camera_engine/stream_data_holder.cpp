#include "stream_data_holder.h"

namespace ganz_camera {

    StreamDataHolder::StreamDataHolder()
        :stop_flag_(true)
        , data_()
        , faces_()
        , processor_()
    {}

    StreamDataHolder::~StreamDataHolder() {
        if (runner_.joinable()) {
            runner_.join();
        }
    }

    void StreamDataHolder::start(EntryProcessFunc func) {
        stop_flag_ = false;
#if 0
        processor_ = func;
        this->process();
#else
        processor_ = func;
        auto runner_func = std::bind(&StreamDataHolder::process, std::move(this));
        runner_ = std::thread(runner_func);
#endif
    }

    void StreamDataHolder::stop() {
        stop_flag_ = true;
#if 0
        if (runner_.joinable()) {
            runner_.join();
        }
#endif
    }

    void StreamDataHolder::put(FrameInfo &&info) {
        data_.push(std::move(info));
    }

    void StreamDataHolder::put(FaceDataVector &&faces) {
        faces_.push(faces);
    }

    void StreamDataHolder::process() {
        while (!stop_flag_) {
            FrameInfo info;
            if (data_.try_pop(info)) {
                FaceDataVector faces;
                faces_.try_pop(faces);
                processor_(*this, info, faces);
            }
        }
    }

}