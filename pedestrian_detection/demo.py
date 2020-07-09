import os
import argparse
from pathlib import Path

import numpy as np
import cv2
import time
from models_api.yolo.yolo_v3_tf import YoloV3TF
from models_api.yolo.yolo5 import YolosV5

# from sort_kalman_filter.sort import Sort
from deep_sort_pytorch.deep_sort import DeepSort


def convert_bboxes_to_xywh(bbox):
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #
    return bbox


def draw_bboxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = (0, 0, 255)
        label = '{} {}'.format("object", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def processing(file, tf_basedir):
    video_stream = cv2.VideoCapture(file)
    # model = YoloV3TF(tf_basedir)
    path = "d:/viktor_project/person_detection/pedestrian_detection/models/yolov5/yolov5m.pt"
    model = YolosV5(path)

    # tracker = Sort(max_age=10)
    model_path = "d:/viktor_project/person_detection/pedestrian_detection/models/deep_sort_2/ckpt.t7"
    deep_sort = DeepSort(model_path)

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        start = time.time()

        # INFO: bboxes must be in the following format: x0,y0,x1,y1
        bboxs, confs = model.inference(frame)
        # bboxs, confs, out_frame = model.inference(frame)
        end = time.time()
        exec_time = end - start
        print(f"detection time: {(1000 * exec_time)} ms")

        start = time.time()
        bboxes = convert_bboxes_to_xywh(bboxs)
        end = time.time()
        exec_time = end - start
        print(f"convert time : {(1000 * exec_time)} ms")

        start = time.time()
        outputs = deep_sort.update(bboxes, confs, frame)

        end = time.time()
        exec_time = end - start
        print(f"deep sort time: {(1000 * exec_time)} ms")

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            detection_frame = draw_bboxes(frame, bbox_xyxy, identities)
        else:
            detection_frame = frame

        # trackers = tracker.update(np.asarray(bboxs), frame)
        # for d in trackers:
        #     p1 = (int(round(d[0])), int(round(d[1])))
        #     p2 = (int(round(d[2])), int(round(d[3])))
        #
        #     cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
        #     track_id = d[4]
        #     cv2.putText(frame, str(track_id), (int(d[0]), int(d[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
        # cv2.imshow("Display", out_frame)
        # cv2.imshow("Display", frame)
        cv2.imshow("Display", detection_frame)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
    video_stream.release()
    # model.finish()


if __name__ == '__main__':

    current_dir = Path(os.getcwd())
    tf_base = current_dir / Path("models/yolo_v3/tensorflow_checkpoint/")
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help="Specify video file for processing")
    args = parser.parse_args()
    if args.file is not None:
        processing(args.file, tf_base)
    else:
        print(f"Usage demo.py --file 'path to video file'")
