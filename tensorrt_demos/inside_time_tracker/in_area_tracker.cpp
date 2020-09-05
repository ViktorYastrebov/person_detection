#include "in_area_tracker.h"
#include "in_area_track.h"

namespace inside_area_tracker {

    using namespace deep_sort;

    InAreaTracker::InAreaTracker(const std::vector<cv::Point> &contour, float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init)
        :Tracker(max_cosine_distance, nn_budget, max_iou_distance, max_age, n_init)
        , contour_(contour)
    {
        build_func_ = std::bind(&InAreaTracker::initialize_track, this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5,
            std::placeholders::_6,
            std::placeholders::_7);
    }

    void InAreaTracker::predict() {
        Tracker::predict();
    }

    void InAreaTracker::update(const common::datatypes::Detections& detections, const std::chrono::milliseconds &time) {
        Tracker::update(detections);
        for (auto track : Tracker::tracks) {
            if (!track->is_confirmed() || track->time_since_update > 1) {
                continue;
            }
            auto bbox = track->to_tlwh();
            cv::Point2f center(bbox(0) + bbox(2) / 2.0f, bbox(1) + bbox(3) / 2.0f);
            auto time_track = std::static_pointer_cast<InAreaTimeTrack>(track);

            bool is_inside = cv::pointPolygonTest(contour_, center, false) > 0;
            time_track->setInside(is_inside);
            (*time_track) += time;
        }
    }

    InAreaTracker::TrackPtr InAreaTracker::initialize_track(
        const common::datatypes::KalmanMeanMatType &mean,
        const common::datatypes::KalmanCovAMatType &covariance,
        int track_id,
        int n_init,
        int max_age,
        const common::datatypes::Feature &feature,
        const int class_id)
    {
        //INFO: make shared give an error under VS 2017: error C2338: You've instantiated std::aligned_storage<Len, Align>....
        return std::shared_ptr<InAreaTimeTrack>(new InAreaTimeTrack(mean, covariance, track_id, n_init, max_age, feature, class_id));
        //return std::shared_ptr<Track>(new Track(mean, covariance, track_id, n_init, max_age, feature, class_id));
    }
}

