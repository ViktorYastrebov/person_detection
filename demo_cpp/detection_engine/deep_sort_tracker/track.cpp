#include "track.h"


Track::Track(KalmanMeanMatType &mean, KalmanCovAMatType &covariance, int track_id, int n_init, int max_age, const Feature &feature)
    :time_since_update(0)
    , track_id(track_id)
    , features()
    , mean(mean)
    , covariance(covariance)
    , hits(1)
    , age(1)
    , _n_init(n_init)
    , _max_age(max_age)
    , state(TrackState::Tentative)
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
    auto pa = kf.update(mean, covariance, detection.to_xyah());
    mean = pa.mean;
    covariance = pa.covariance;

    featuresAppendOne(detection.feature);
    hits += 1;
    time_since_update = 0;
    if (state == TrackState::Tentative && hits >= _n_init) {
        state = TrackState::Confirmed;
    }
}

void Track::mark_missed() {
    if (state == TrackState::Tentative) {
        state = TrackState::Deleted;
    } else if (time_since_update > _max_age) {
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

DetectionBox Track::to_tlwh() const {
    DetectionBox ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2) / 2);
    return ret;
}

//INFO: might need to refactor
void Track::featuresAppendOne(const Feature& feature) {
    auto size = features.rows();
    Features newfeatures = Features(size + 1, 512);
    newfeatures.block(0, 0, size, 512) = features;
    newfeatures.row(size) = feature;
    features = newfeatures;
}