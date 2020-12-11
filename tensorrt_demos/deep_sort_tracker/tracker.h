#pragma once

#include "common/datatypes.h"
#include "base_tracker.h"

namespace deep_sort {

    class TrackerImpl;

    class DEEP_SORT_TRACKER Tracker : public AbstractTracker {
    public:
        Tracker(float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3,
            const int max_time_since_update = 3, const int max_artificial_updates = 30);
        virtual ~Tracker();
        void predict() override;
        void update(const common::datatypes::Detections& detections) override;
        const std::vector<TrackPtr> &getTracks() const override;
    protected:
        TrackerImpl *impl_;
    };

}