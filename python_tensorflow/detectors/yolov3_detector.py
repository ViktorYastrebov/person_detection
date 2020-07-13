from pathlib import Path
from .model import yolov3
from .nms_utils import gpu_nms
import tensorflow as tf
import cv2
import numpy as np


class YoloV3:
    def __init__(self, base_dir: Path):
        self.size = 416
        self.base_dir = base_dir
        self.anchors = self.base_dir / "yolo_anchors.txt"
        self.class_file = self.base_dir / "coco.names"
        self.classes = self._read_class_names(self.class_file)
        self.class_num = len(self.classes)
        self.session = tf.Session()

        self.input_data = tf.placeholder(tf.float32, [1, self.size, self.size, 3], name='input_data')

        yolo_model = yolov3(self.class_num, self._parse_anchors(self.anchors))
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(self.input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        self.boxes, self.scores, self.labels = gpu_nms(pred_boxes,
                                                       pred_scores,
                                                       self.class_num,
                                                       max_boxes=200,
                                                       score_thresh=0.3,
                                                       nms_thresh=0.45)

        saver = tf.train.Saver()
        checkpoint = base_dir / "yolov3.ckpt"
        saver.restore(self.session, str(checkpoint))

    def inference(self, frame):
        height_ori, width_ori = frame.shape[:2]
        img = cv2.resize(frame, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_ = self.session.run([self.boxes, self.scores, self.labels],
                                                    feed_dict={self.input_data: img})
        # rescale the coordinates to the original image
        boxes_[:, [0, 2]] *= (width_ori / float(self.size))
        boxes_[:, [1, 3]] *= (height_ori / float(self.size))

        filtered_scores = list()
        # inference_outputs = list()
        normalized_bboxes = list()
        for i in range(len(boxes_)):
            if scores_[i] > 0.5:
                x0, y0, x1, y1 = boxes_[i]
                normalized_bboxes.append((x0, y0, x1, y1))
                # inference_outputs.append(ModelOutput(x0, y0, x1 - x0, y1 - y0, scores_[i], labels_[i]))
                # cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                filtered_scores.append(scores_[i])
        return normalized_bboxes, filtered_scores

    def finish(self):
        self.session.close()

    def _parse_anchors(self, anchor_path):
        anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
        return anchors

    def _read_class_names(self, file: Path):
        names = {}
        with file.open('r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names
