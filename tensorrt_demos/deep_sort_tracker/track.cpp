#include "track.h"
#include "kalman_filter.h"


namespace deep_sort {
    using namespace common::datatypes;

    Track::Track(const KalmanMeanMatType &mean, const KalmanCovAMatType &covariance, int track_id, int n_init, int max_age, const Feature &feature,
        const int class_id,
        const int max_time_since_update,
        const int max_artificial_updates
    )
        :time_since_update(0)
        , track_id(track_id)
        , class_id(class_id)
        , features()
        , mean(mean)
        , covariance(covariance)
        , hits(1)
        , age(1)
        , _n_init(n_init)
        , _max_age(max_age)
        , state(TrackState::Tentative)
        , artificial_updates_(0)
        , max_time_since_update_(max_time_since_update)
        , max_artificial_updates_(max_artificial_updates)
    {
        features = Features(1, FEATURES_SIZE);
        features.row(0) = feature;
    }

    void Track::predit(KalmanFilter &kf) {
        kf.predict(mean, covariance);
        age += 1;
        time_since_update += 1;
    }

    void Track::update(KalmanFilter & kf, const Detection &detection) {
        artificial_updates_ = 0;
        auto pa = kf.update(mean, covariance, detection.to_xyah());
        mean = pa.mean;
        covariance = pa.covariance;

        featuresAppendOne(detection.feature);
        hits += 1;
        time_since_update = 0;
        if (state == TrackState::Tentative && hits >= _n_init) {
            state = TrackState::Confirmed;
        }

        //INFO: hold the history
        auto state_value = to_tlwh();
        last_states_.push_back(state_value);
        if (last_states_.size() >= QUEUE_SIZE) {
            last_states_.pop_front();
        }
    }

    void Track::artificial_update(KalmanFilter &kf) {
        auto state = to_tlwh();
        if (last_states_.size() > 1) {
            auto first_state = last_states_.front();

            //INFO: find center points and find diff from them
            float vx = (last_states_[1].position(0) + last_states_[1].position(2) / 2)  - (last_states_[0].position(0) + last_states_[0].position(2) / 2);
            float vy = (last_states_[1].position(1) + last_states_[1].position(3) / 2) - (last_states_[0].position(1) + last_states_[0].position(3) / 2);
            float first_len = std::sqrt(vx *vx  + vy * vy);
            float res_vx = vx;
            float res_vy = vy;
            for (int i = 2; i < last_states_.size(); ++i) {
                res_vx += ( (last_states_[i].position(0) + last_states_[i].position(2) / 2)  - (last_states_[i - 1].position(0) + last_states_[i - 1].position(2) / 2) );
                res_vy += ( (last_states_[i].position(1) + last_states_[i].position(3) / 2) -  (last_states_[i - 1].position(1) + last_states_[i - 1].position(3) / 2) );
            }

            //for (const auto &it : last_states_) {
            //    auto vx_it = it.velocity(0) + it.velocity(2) / 2.0f;
            //    auto vy_it = it.velocity(1) + it.velocity(3) / 2.0f;
            //    res_vx += vx_it;
            //    res_vy += vy_it;
            //}
            float res_len = std::sqrt(res_vx*res_vx + res_vy * res_vy);
            res_vx = (res_vx / res_len) * first_len * 0.01f;
            res_vy = (res_vy / res_len) * first_len * 0.01f;

            DetectionBox ret(
                state.position(0) + res_vx,
                state.position(1) + res_vy,
                state.position(2),
                state.position(3)
            );

            ret(0, 0) += (ret(0, 2)*0.5f);
            ret(0, 1) += (ret(0, 3)*0.5f);
            ret(0, 2) /= ret(0, 3);

            auto pa = kf.update(mean, covariance, ret);
            mean = pa.mean;
            covariance = pa.covariance;
        } else {
            kf.predict(mean, covariance);
        }

        if (artificial_updates_ > max_artificial_updates_) {
            set_deleted();
        }

        ++artificial_updates_;
    }


    Track::TrackType Track::getType() const {
        return TrackType::DEFAULT;
    }

    void Track::mark_missed() {
        if (state == TrackState::Tentative && time_since_update > max_time_since_update_) {
            state = TrackState::Deleted;
        } else if (time_since_update > max_artificial_updates_) {
            state = TrackState::Deleted;
        }
    }

    bool Track::is_confirmed() const {
        return state == TrackState::Confirmed;
    }

    bool Track::is_deleted() const {
        return state == TrackState::Deleted;
    }

    bool Track::is_tentative() const {
        return state == TrackState::Tentative;
    }

    void Track::set_deleted() {
        state = TrackState::Deleted;
    }

    Track::State Track::to_tlwh() const {
        DetectionBox pos = mean.leftCols(4);
        pos(2) *= pos(3);
        pos.leftCols(2) -= (pos.rightCols(2) / 2);

        DetectionBox vel = mean.rightCols(4);
        vel(2) *= vel(3);
        vel.leftCols(2) -= (vel.rightCols(2) / 2);
        return { pos, vel };
    }

    //INFO: might need to refactor
    void Track::featuresAppendOne(const Feature& feature) {
        auto size = features.rows();
        Features newfeatures = Features(size + 1, FEATURES_SIZE);
        newfeatures.block(0, 0, size, FEATURES_SIZE) = features;
        newfeatures.row(size) = feature;
        features = newfeatures;
    }
}