import os
import argparse
from pathlib import Path

import numpy as np
import cv2
import time
from models_api.yolo.yolo_v3_tf import YoloV3TF

from deep_sort import nn_matching
from deep_sort.encoder import create_box_encoder
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.preprocessing import non_max_suppression

from sort_kalman_filter.sort import Sort


def processing(file, tf_basedir, deep_sort_model):
    video_stream = cv2.VideoCapture(file)
    # base_dir =  Path("d:/viktor_project/person_detection/pedestrian_detection/models/yolo_v3/tensorflow_checkpoint/")
    model = YoloV3TF(tf_basedir)

    # Deep Sort

    # max_cosine_distance = 0.3
    # nn_budget = None
    # nms_max_overlap = 1.0

    # Deep SORT
    # model_filename = "d:/viktor_project/person_detection/pedestrian_detection/models/deep_sort/mars-small128.pb"
    # encoder = create_box_encoder(str(deep_sort_model), batch_size=1)

    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # tracker = Tracker(metric)

    tracker = Sort(max_age=10)

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        start = time.time()
        # INFO: bboxes must be in the following format: x,y,w,h
        bboxs, confs, out_frame = model.inference(frame)

        end = time.time()
        exec_time = end - start
        print(f"detection time: {(1000 * exec_time)} ms")

        trackers = tracker.update(np.asarray(bboxs), frame)

        for d in trackers:
            p1 = (int(round(d[0])), int(round(d[1])))
            p2 = (int(round(d[2])), int(round(d[3])))

            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
            track_id = d[4]
            cv2.putText(frame, str(track_id), (int(d[0]), int(d[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        ##################################
        # features = encoder(frame, bboxs)
        # detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in
        #               zip(bboxs, confs, features)]
        #
        # # Run non-maxima suppression.
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = non_max_suppression(boxes, nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]
        #
        # # Call the tracker
        # start = time.time()
        # tracker.predict()
        # tracker.update(detections)
        # end = time.time()
        # exec_time = end - start
        # print(f"track time: {(1000 * exec_time)} ms")

        # for track in tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue
        #     bbox = track.to_tlbr()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        #     cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        # for det in detections:
        #     bbox = det.to_tlbr()
        #     score = "%.2f" % round(det.confidence * 100, 2)
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        #     cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0, 255, 0), 2)
        ##################################

        # end = time.time()
        # exec_time = end - start
        # print(f"total time: {(1000 * exec_time)} ms")

        cv2.imshow("Display", frame)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
    video_stream.release()
    model.finish()


if __name__ == '__main__':

    current_dir = Path(os.getcwd())
    tf_base = current_dir / Path("models/yolo_v3/tensorflow_checkpoint/")
    tracker_base = current_dir / Path("models/deep_sort/mars-small128.pb")

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help="Specify video file for processing")
    args = parser.parse_args()
    if args.file is not None:
        processing(args.file, tf_base, tracker_base)
    else:
        print(f"Usage demo.py --file 'path to video file'")
