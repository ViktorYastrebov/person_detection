import os
import argparse
from pathlib import Path

import numpy as np
import cv2
import time
from models_api.yolo.yolo_v3_tf import YoloV3TF

from sort_kalman_filter.sort import Sort


def processing(file, tf_basedir):
    video_stream = cv2.VideoCapture(file)
    model = YoloV3TF(tf_basedir)
    tracker = Sort(max_age=10)

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        start = time.time()
        # INFO: bboxes must be in the following format: x0,y0,x1,y1
        bboxs, confs, out_frame = model.inference(frame)
        trackers = tracker.update(np.asarray(bboxs), frame)
        for d in trackers:
            p1 = (int(round(d[0])), int(round(d[1])))
            p2 = (int(round(d[2])), int(round(d[3])))

            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
            track_id = d[4]
            cv2.putText(frame, str(track_id), (int(d[0]), int(d[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        end = time.time()
        exec_time = end - start
        print(f"detection time: {(1000 * exec_time)} ms")

        cv2.imshow("Display", frame)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
    video_stream.release()
    model.finish()


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
