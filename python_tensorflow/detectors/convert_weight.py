# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function
import tensorflow as tf

from detectors.model import yolov3
from detectors.misc_utils import parse_anchors, load_weights
import shutil
from pathlib import Path

def convert():
    num_class = 80
    img_size = 416
    weight_path = '../models/yolov3_darknet_weights/yolov3.weights'
    save_path = '../models/yolov3_checkpoint/yolov3.ckpt'
    classes_file = '../models/yolov3_darknet_weights/coco.names'
    anchors_file = '../models/yolov3_darknet_weights/yolo_anchors.txt'

    anchors = parse_anchors(anchors_file)

    model = yolov3(80, anchors)
    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

        with tf.variable_scope('yolov3'):
            feature_map = model.forward(inputs)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

        load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
        sess.run(load_ops)
        saver.save(sess, save_path=save_path)
        print('TensorFlow model checkpoint has been saved to {}'.format(save_path))
    base_dir = Path(save_path).parent
    shutil.copy(classes_file, str(base_dir))
    shutil.copy(anchors_file, str(base_dir))
    print(f"Files {classes_file}, {anchors_file}  copied to : {base_dir}")


if __name__ == '__main__':
    convert()


