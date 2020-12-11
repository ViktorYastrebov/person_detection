#pragma once
#include <memory>

#include "deep_sort_types.h"
#include "decl_spec.h"
#include "track.h"


namespace deep_sort {
    class DEEP_SORT_TRACKER AbstractTracker {
    public:
        using TrackPtr = std::shared_ptr<Track>;

        virtual ~AbstractTracker() = 0;
        virtual void predict() = 0;
        virtual void update(const common::datatypes::Detections& detections) = 0;
        virtual const std::vector<TrackPtr> &getTracks() const = 0;
    };
}
