import os
import argparse
from pathlib import Path

import cv2
import time
from models_api.yolo5 import YolosV5
from deep_sort.deep_sort import DeepSort


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
        label = '{} {}'.format("person", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def process_detections(file, name):
    video_stream = cv2.VideoCapture(file)
    path = Path(os.getcwd()) / "models" / "yolov5" / name
    model = YolosV5(path)

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        start = time.time()
        # INFO: bboxes are returned be in the following format: x0,y0,x1,y1
        bboxs, confs = model.inference(frame)
        end = time.time()
        exec_time = end - start

        for box in bboxs:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        print(f"detection time: {(1000 * exec_time)} ms")
        cv2.imshow("Display", frame)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
    video_stream.release()


def process_detections_with_deep_sort(file, name, max_iou_distance=0.7, max_age=30):
    """
    INFO: here is the problem with the DeepSort. It's processing without batching
    """
    video_stream = cv2.VideoCapture(file)
    base_dir = Path(os.getcwd())
    path = base_dir / "models" / "yolov5" / name
    model = YolosV5(path)

    deep_sort_path = base_dir / "models" / "deep_sort" / "ckpt.t7"
    deep_sort = DeepSort(deep_sort_path, max_iou_distance, max_age)

    def convert_bboxes_to_xywh(bbox):
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #
        return bbox

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        start = time.time()

        # INFO: bboxes must be in the following format: x0,y0,x1,y1
        bboxs, confs = model.inference(frame, [0])
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
        cv2.imshow("Display", detection_frame)
        ret = cv2.waitKey(1)
        if ret == 27:
            break
    video_stream.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help="Specify video file for processing")
    models = "[yolov3-spp.pt, yolov5s.py, yolov5m.py, yolov5l.pt, yolov5x.pt]"
    parser.add_argument('--name', default='yolov5m.pt', help=f'model names from set:{models}')
    parser.add_argument('--iou', default=0.7, help='max iou distance')
    parser.add_argument('--age', default=30, help='Maximum number of frames of misses before a track is deleted.')
    args = parser.parse_args()
    if args.file is not None and args.name is not None:
        # process_detections(args.file, args.name)
        process_detections_with_deep_sort(args.file, args.name)
    else:
        print(f"Usage demo.py --file 'path to video file' --name 'model_name'")
