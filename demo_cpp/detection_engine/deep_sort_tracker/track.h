#pragma once

#include "common.h"
#include "kalman_filter.h"

class Track
{
public:
    enum TrackState {
        Tentative = 1,
        Confirmed,
        Deleted
    };

    struct DetectionRow {
        cv::Rect box;
        float confidence;
        Feature feature;
        //DETECTBOX to_xyah() const;
        //DETECTBOX to_tlbr() const;
    };

    Track(KalmanMeanMatType &mean, KalmanCovAMatType &covariance, int track_id, int n_init, int max_age, const Feature &feature);
    ~Track() = default;

    void predit(KalmanFilter &kf);
    void update(KalmanFilter &kf, const DetectionRow &detection);

    void mark_missed();
    bool is_confirmed() const;
    bool is_deleted() const;
    bool is_tentative() const;


    //TODO: refactor do not use python style in C++
    //DETECTBOX to_tlwh();
    int time_since_update;
    int track_id;
    Features features;
    KalmanMeanMatType mean;
    KalmanCovAMatType covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;
private:
    void featuresAppendOne(const Feature& feature);
};
