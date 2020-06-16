# coding: utf-8

# from __future__ import division, print_function
from pathlib import Path

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
import os

from models_api.yolo.yolov3_tensorflow.misc_utils import parse_anchors, read_class_names
from models_api.yolo.yolov3_tensorflow.nms_utils import gpu_nms
from models_api.yolo.yolov3_tensorflow.plot_utils import plot_one_box

from models_api.yolo.yolov3_tensorflow.model import yolov3


def process():
    parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
    parser.add_argument("input_video", type=str,
                        help="The path of the input video.")
    # parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
    #                     help="The path of the anchor txt file.")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
    # parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
    #                     help="The path of the class names.")
    # parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
    #                     help="The path of the weights to restore.")
    args = parser.parse_args()

    print(f"current dir: {os.getcwd()}")

    base_dir = Path("d:/viktor_project/person_detection/pedestrian_detection/models/yolo_v3/tensorflow_checkpoint/")
    print(f"exists : {base_dir.exists()}")

    anchors = parse_anchors(str(base_dir / "yolo_anchors.txt"))
    classes = read_class_names(str(base_dir / "coco.names"))
    num_class = len(classes)

    vid = cv2.VideoCapture(args.input_video)

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.3,
                                        nms_thresh=0.45)

        saver = tf.train.Saver()
        checkpoint = base_dir / "yolov3.ckpt"
        saver.restore(sess, str(checkpoint))

        while True:
            ret, img_ori = vid.read()
            if not ret:
                break
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            start_time = time.time()
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            end_time = time.time()
            exec_time = end_time - start_time
            print(f"time: {(1000 * exec_time)} ms")

            # rescale the coordinates to the original image
            boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))

            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                # plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i]
                # * 100), color=color_table[labels_[i]])
                plot_one_box(img_ori, [x0, y0, x1, y1])
            # cv2.putText(img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
            #             ontScale=1, color=(0, 255, 0), thickness=2)
            cv2.imshow('image', img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()


if __name__ == '__main__':
    process()
