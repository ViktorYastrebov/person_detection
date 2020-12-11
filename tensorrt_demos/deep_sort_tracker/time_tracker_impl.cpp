#include "time_tracker_impl.h"
#include "time_track.h"

namespace deep_sort {

    TimeTrackerImpl::TimeTrackerImpl(const std::vector<cv::Point> &contour, float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init,
        const int max_time_since_update, const int max_artificial_updates)
        :TrackerImpl(max_cosine_distance, nn_budget, max_iou_distance, max_age, n_init, max_time_since_update, max_artificial_updates)
        , contour_(contour)
    {
        build_func_ = std::bind(&TimeTrackerImpl::initialize_track, this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5,
            std::placeholders::_6,
            std::placeholders::_7);
    }

    void TimeTrackerImpl::predict() {
        TrackerImpl::predict();
    }

    void TimeTrackerImpl::update(const common::datatypes::Detections& detections, const std::chrono::milliseconds &time) {
        TrackerImpl::update(detections);
        for (auto track : TrackerImpl::tracks) {
            if (!track->is_confirmed() || track->time_since_update > 1) {
                continue;
            }
            //INFO: can be optimized, just get the centers from the mean
            auto rets = track->to_tlwh();
            cv::Point2f center(rets.position(0) + rets.position(2) / 2.0f, rets.position(1) + rets.position(3) / 2.0f);
            auto time_track = std::static_pointer_cast<TimeTrack>(track);

            bool is_inside = cv::pointPolygonTest(contour_, center, false) > 0;
            time_track->setInside(is_inside);
            (*time_track) += time;
        }
    }

    AbstractTracker::TrackPtr TimeTrackerImpl::initialize_track(
        const common::datatypes::KalmanMeanMatType &mean,
        const common::datatypes::KalmanCovAMatType &covariance,
        int track_id,
        int n_init,
        int max_age,
        const common::datatypes::Feature &feature,
        const int class_id)
    {
        //INFO: make shared give an error under VS 2017: error C2338: You've instantiated std::aligned_storage<Len, Align>....
        return std::shared_ptr<TimeTrack>(new TimeTrack(mean, covariance, track_id, n_init, max_age, feature, class_id,
            max_time_since_update_, max_artificial_updates_));
        //return std::shared_ptr<Track>(new Track(mean, covariance, track_id, n_init, max_age, feature, class_id));
    }
}

