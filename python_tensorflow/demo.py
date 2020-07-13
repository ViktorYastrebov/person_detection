# Adapted from: https://github.com/Qidian213/deep_sort_yolov3.git

import os
import argparse
import numpy as np
from pathlib import Path

import cv2
import time
from detectors.yolov3_detector import YoloV3
from sort_kalman_filter.sort import Sort
from deep_sort.encoder import create_box_encoder
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.preprocessing import non_max_suppression


def process_detections(file, tf_basedir):
    video_stream = cv2.VideoCapture(file)
    model = YoloV3(tf_basedir)

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        start = time.time()
        bboxs, confs = model.inference(frame)
        for box in bboxs:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        end = time.time()
        exec_time = end - start
        print(f"detection time: {(1000 * exec_time)} ms")

        cv2.imshow("Display", frame)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
    video_stream.release()
    model.finish()


def process_detection_with_kalman_filter(file, tf_basedir):
    video_stream = cv2.VideoCapture(file)
    model = YoloV3(tf_basedir)
    tracker = Sort(max_age=10)

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        start = time.time()
        bboxs, confs = model.inference(frame)
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
        cv2.imshow("Display", frame)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
    video_stream.release()
    model.finish()


def processing_detection_with_deep_sort(file, tf_basedir, deep_sort_model):
    video_stream = cv2.VideoCapture(file)
    model = YoloV3(tf_basedir)

    # DEEP SORT params
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    encoder = create_box_encoder(str(deep_sort_model), batch_size=1)
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    def convert_to_xywh(bboxes:np.ndarray):
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        start = time.time()
        # here there must be bounding boxes in format (x, y, width, height)
        bboxs, confs = model.inference(frame)
        end = time.time()
        exec_time = end - start
        print(f"detection time: {(1000 * exec_time)} ms")

        bboxes_converted = convert_to_xywh(np.asarray(bboxs))

        features = encoder(frame, bboxes_converted)

        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in
                      zip(bboxes_converted, confs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0, 255, 0), 2)

        cv2.imshow("Display", frame)
        ret = cv2.waitKey(1)
        if ret == 27:
            break

    video_stream.release()
    model.finish()


if __name__ == '__main__':
    current_dir = Path(os.getcwd())
    tf_base = current_dir / Path("models/yolov3_checkpoint")
    deep_sort = current_dir / Path("models/deep_sort/mars-small128.pb")
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help="Specify video file for processing")
    args = parser.parse_args()
    if args.file is not None:
        # process_detections(args.file, tf_base)
        # process_detection_with_kalman_filter(args.file, tf_base)
        processing_detection_with_deep_sort(args.file, tf_base, deep_sort)
    else:
        print(f"Usage demo.py --file 'path to video file'")
