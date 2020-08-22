#include "tracker.h"
#include "linear_assignment.h"
#include "nn_matching.h"
#include <vector>


namespace deep_sort {

    using namespace common::datatypes;


    Tracker::Tracker(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init)
        :metric_processor(std::make_unique<NearNeighborDisMetric>(NearNeighborDisMetric::METRIC_TYPE::cosine, max_cosine_distance, nn_budget))
        , max_iou_distance(max_iou_distance)
        , max_age(max_age)
        , n_init(n_init)
        , _next_idx(1)
        , kalman_filter(std::make_unique<KalmanFilter>())
        , tracks()
    {}

    void Tracker::predict() {
        for (Track& track : tracks) {
            track.predit(*kalman_filter);
        }
    }

    void Tracker::update(const Detections& detections)
    {
        TrackerMatch res;
        match(detections, res);

        std::vector<MatchData>& matches = res.matches;
        for (MatchData& data : matches) {
            int track_idx = data.first;
            int detection_idx = data.second;
            tracks[track_idx].update(*kalman_filter, detections[detection_idx]);
        }
        std::vector<int>& unmatched_tracks = res.unmatched_tracks;
        for (int& track_idx : unmatched_tracks) {
            this->tracks[track_idx].mark_missed();
        }
        std::vector<int>& unmatched_detections = res.unmatched_detections;
        for (int& detection_idx : unmatched_detections) {
            initiate_track(detections[detection_idx]);
        }
        std::vector<Track>::iterator it = tracks.begin();
        for (it; it != tracks.end();) {
            if ((*it).is_deleted()) {
                it = tracks.erase(it);
            } else {
                ++it;
            }
        }
        std::vector<int> active_targets;
        std::vector<TrackerResult> tid_features;
        for (Track& track : tracks) {
            if (!track.is_confirmed()) {
                continue;
            }
            active_targets.push_back(track.track_id);
            tid_features.push_back(std::make_pair(track.track_id, track.features));
            Features t = Features(0, FEATURES_SIZE);
            track.features = t;
        }
        metric_processor->partial_fit(tid_features, active_targets);
    }


    CostMatrixType Tracker::gated_metric(
        std::vector<Track>& tracks,
        const common::datatypes::Detections& dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices
    )
    {
        Features features(detection_indices.size(), FEATURES_SIZE);
        int pos = 0;
        for (int i : detection_indices) {
            features.row(pos++) = dets[i].feature;
        }

        std::vector<int> targets;
        for (int i : track_indices) {
            targets.push_back(tracks[i].track_id);
        }
        CostMatrixType cost_matrix = metric_processor->distance(features, targets);

        CostMatrixType res = linear_assignment::gate_cost_matrix(kalman_filter.get(), cost_matrix, tracks, dets, track_indices, detection_indices);
        return res;
    }

    CostMatrixType Tracker::iou_cost(
        std::vector<Track>& tracks,
        const Detections& dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices)
    {

        //!!!python diff: track_indices && detection_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }
    //    if(detection_indices.empty() == true) {
    //        for(size_t i = 0; i < dets.size(); i++) {
    //            detection_indices.push_back(i);
    //        }
    //    }
        int rows = track_indices.size();
        int cols = detection_indices.size();
        CostMatrixType cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
        for (int i = 0; i < rows; i++) {
            int track_idx = track_indices[i];
            if (tracks[track_idx].time_since_update > 1) {
                cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
                continue;
            }
            auto bbox = tracks[track_idx].to_tlwh();
            int csize = detection_indices.size();
            DetectionBoxes candidates(csize, 4);
            for (int k = 0; k < csize; k++) {
                candidates.row(k) = dets[detection_indices[k]].tlwh;
            }
            Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
            cost_matrix.row(i) = rowV;
        }
        return cost_matrix;
    }

    Eigen::VectorXf Tracker::iou(DetectionBox& bbox, DetectionBoxes &candidates)
    {
        float bbox_tl_1 = bbox[0];
        float bbox_tl_2 = bbox[1];
        float bbox_br_1 = bbox[0] + bbox[2];
        float bbox_br_2 = bbox[1] + bbox[3];
        float area_bbox = bbox[2] * bbox[3];

        Eigen::Matrix<float, -1, 2> candidates_tl;
        Eigen::Matrix<float, -1, 2> candidates_br;
        candidates_tl = candidates.leftCols(2);
        candidates_br = candidates.rightCols(2) + candidates_tl;

        int size = int(candidates.rows());
        Eigen::VectorXf res(size);
        for (int i = 0; i < size; i++) {
            float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
            float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
            float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
            float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

            float w = br_1 - tl_1; w = (w < 0 ? 0 : w);
            float h = br_2 - tl_2; h = (h < 0 ? 0 : h);
            float area_intersection = w * h;
            float area_candidates = candidates(i, 2) * candidates(i, 3);
            res[i] = area_intersection / (area_bbox + area_candidates - area_intersection);
        }
        return res;
    }

    void Tracker::match(const Detections& detections, TrackerMatch& res) {
        std::vector<int> confirmed_tracks;
        std::vector<int> unconfirmed_tracks;
        int idx = 0;
        for (Track& t : tracks) {
            if (t.is_confirmed()) confirmed_tracks.push_back(idx);
            else unconfirmed_tracks.push_back(idx);
            idx++;
        }

        TrackerMatch matcha = linear_assignment::matching_cascade(
            this,
            &Tracker::gated_metric,
            metric_processor->getMatingThreshold(),
            max_age,
            tracks,
            detections,
            confirmed_tracks);

        std::vector<int> iou_track_candidates;
        iou_track_candidates.assign(unconfirmed_tracks.begin(), unconfirmed_tracks.end());
        std::vector<int>::iterator it;
        for (it = matcha.unmatched_tracks.begin(); it != matcha.unmatched_tracks.end();) {
            int idx = *it;
            if (tracks[idx].time_since_update == 1) { //push into unconfirmed
                iou_track_candidates.push_back(idx);
                it = matcha.unmatched_tracks.erase(it);
                continue;
            }
            ++it;
        }
        TrackerMatch matchb = linear_assignment::min_cost_matching(
            this,
            &Tracker::iou_cost,
            max_iou_distance,
            this->tracks,
            detections,
            iou_track_candidates,
            matcha.unmatched_detections);
        //INFO: get result
        res.matches.assign(matcha.matches.begin(), matcha.matches.end());
        res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());
        //INFO: unmatched_tracks
        res.unmatched_tracks.assign(
            matcha.unmatched_tracks.begin(),
            matcha.unmatched_tracks.end());
        res.unmatched_tracks.insert(
            res.unmatched_tracks.end(),
            matchb.unmatched_tracks.begin(),
            matchb.unmatched_tracks.end());
        res.unmatched_detections.assign(
            matchb.unmatched_detections.begin(),
            matchb.unmatched_detections.end());
    }

    void Tracker::initiate_track(const Detection& detection) {
        auto data = kalman_filter->initiate(detection.to_xyah());
        auto mean = data.mean;
        auto covariance = data.covariance;

        tracks.push_back(Track(mean, covariance, _next_idx, n_init, max_age, detection.feature, detection.class_id));
        _next_idx += 1;
    }

    const std::vector<Track> &Tracker::getTracks() const {
        return tracks;
    }
}