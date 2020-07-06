#include "track_processor.h"


TrackProcessor::TrackProcessor(const int idx, CentralWidget &cw)
    : idx_(idx)
    , central_(cw)
    , stop_flag_(true)
    , tracks_(10)
{}

TrackProcessor::~TrackProcessor()
{
    if (runner_.joinable()) {
        runner_.join();
    }
}

void TrackProcessor::start() {
    stop_flag_ = false;
    auto func = std::bind(&TrackProcessor::processingImpl, this);
    runner_ = std::thread(func);
}

void TrackProcessor::stop() {
    stop_flag_ = true;
}

void TrackProcessor::put(TrackInputData &&input) {
    detections_.push(input);
}

int TrackProcessor::getId() const {
    return idx_;
}

void TrackProcessor::processingImpl() {
    while (!stop_flag_) {
        TrackInputData data;
        if (detections_.try_pop(data)) {
            if (data.detections.empty()) {
                continue;
            }
            auto tracks = tracks_.update(data.detections);
            central_.putTo(idx_, data.frame, tracks);
        }
    }
}