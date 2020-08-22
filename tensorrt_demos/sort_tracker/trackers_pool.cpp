#include "trackers_pool.h"

#include "Hungarian.h"
#include <set>

namespace tracker {

    using namespace common::datatypes;

    namespace helpers {
        double GetIOU(const cv::Rect &bb_test, const cv::Rect &bb_gt) {
            float in = static_cast<float>((bb_test & bb_gt).area());
            float un = bb_test.area() + bb_gt.area() - in;
            //TODO: check for close function
            if (un < DBL_EPSILON)
                return 0.0;
            return (double)(in / un);
        }

        double iou(const DetectionBox &lbox, const DetectionBox &rbox) {
            float interBox[] = {
                std::max(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
                std::min(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
                std::max(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
                std::min(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
            };

            if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
                return 0.0f;

            float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
            return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
        }
    }

    TrackersPool::TrackersPool(int max_age, int min_hits)
        :initialized_(false)
        , max_age_(max_age)
        , min_hits_(min_hits)
    {}

    std::vector<TrackResult> TrackersPool::update(const common::datatypes::DetectionResults &detections) {
        ++frame_counter_;

        if (!initialized_) {
            std::vector<TrackResult> out;
            for (const auto &detection : detections) {
                //INFO: test
                cv::Rect cv_rect(detection.bbox(0), detection.bbox(1), detection.bbox(2), detection.bbox(3));
                trackers_.push_back(KalmanTracker(detection.bbox, detection.class_id));
                out.push_back({ cv_rect, trackers_.back().getID(), detection.class_id });
            }
            initialized_ = true;
            return out;
        }

        //std::vector<cv::Rect> predicted_boxes;
        std::vector<DetectionBox> predicted_boxes;
        for (auto it = trackers_.begin(); it != trackers_.end();)
        {
            //cv::Rect pBox = (*it).predict();
            //if (pBox.x >= 0 && pBox.y >= 0) {
            //INFO: test
            auto rect = (*it).predict();
            if(rect(0) >= 0.0f && rect(1) >= 0.0f) {
                //predicted_boxes.push_back(pBox);
                predicted_boxes.push_back(rect);
                ++it;
            } else {
                it = trackers_.erase(it);
            }
        }
        std::size_t n_predicted = predicted_boxes.size();
        std::size_t n_detected = detections.size();

        std::vector<std::vector<double>> iouMatrix(n_predicted, std::vector<double>(n_detected, 0));


        for (unsigned int i = 0; i < n_predicted; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < n_detected; j++)
            {
                iouMatrix[i][j] = 1.0 - helpers::iou(predicted_boxes[i], detections[j].bbox);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        std::vector<int> assignment;
        HungAlgo.Solve(iouMatrix, assignment);

        std::set<int> allItems;
        std::set<int> matchedItems;
        std::set<int> unmatchedDetections;
        std::set<int> unmatchedTrajectories;
        std::vector<cv::Point> matchedPairs;

        // INFO: there are unmatched detections
        if (n_detected > n_predicted) {
            for (unsigned int n = 0; n < n_detected; ++n) {
                allItems.insert(n);
            }

            for (unsigned int i = 0; i < n_predicted; ++i) {
                matchedItems.insert(assignment[i]);
            }

            set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(),
                            std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        } else {
            // there are unmatched trajectory/predictions
            if (n_detected < n_predicted) {
                for (unsigned int i = 0; i < n_predicted; ++i) {
                    // unassigned label will be set as -1 in the assignment algorithm
                    if (assignment[i] == -1) {
                        unmatchedTrajectories.insert(i);
                    }
                }
            }
        }

        for (unsigned int i = 0; i < n_predicted; ++i) {
            // pass over invalid values
            if (assignment[i] == -1) {
                continue;
            }
            if (1 - iouMatrix[i][assignment[i]] < iou_threshold_) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            } else {
                matchedPairs.push_back(cv::Point(i, assignment[i]));
            }
        }

        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); ++i) {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            //INFO: test
            cv::Rect cv_rect(detections[detIdx].bbox(0), detections[detIdx].bbox(1), detections[detIdx].bbox(2), detections[detIdx].bbox(3));
            trackers_[trkIdx].update(cv_rect);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections) {
            //KalmanTracker tracker = KalmanTracker(detections[umd]);
            //trackers.push_back(tracker);
            trackers_.push_back(KalmanTracker(detections[umd].bbox, detections[umd].class_id));
        }

        std::vector< TrackResult > results;
        for (auto it = trackers_.begin(); it != trackers_.end();) {
            if (((*it).getTimeSinceUpdate() < 1) && (it->getHitSteak() >= min_hits_ || frame_counter_ <= min_hits_)) {
                //INFO: test
                auto rect = it->getState();
                cv::Rect cv_rect(rect(0), rect(1), rect(2), rect(3));
                results.push_back({ cv_rect , it->getID(), it->getClassID() });
                ++it;
            } else {
                ++it;
            }
            // remove dead tracklet
            if (it != trackers_.end() && it->getTimeSinceUpdate() > max_age_) {
                it = trackers_.erase(it);
            }
        }
        return results;
    }

}