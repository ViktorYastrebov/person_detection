import argparse
import sys
import cv2
import time
import numpy as np
import utils as utils
import tensorflow as tf


def run_processing(video_stream: cv2.VideoCapture):

    return_elements = ["input/input_data:0",
                       "pred_sbbox/concat_2:0",
                       "pred_mbbox/concat_2:0",
                       "pred_lbbox/concat_2:0"
                       ]
    pb_file = "./model/model.pb"
    num_classes = 80
    input_size = 416
    graph = tf.Graph()
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    with tf.Session(graph=graph) as sess:
        while True:
            return_value, frame = video_stream.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.7)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            result_image = utils.draw_bbox(frame, bboxes)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            info = "time: %.2f ms" % (1000 * exec_time)
            print(info)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--file', type=str, help='Specify video file for processing')
    parser.add_argument('--stream', type=int, help='Specify video stream for processing')

    args = parser.parse_args()
    if args.file is not None:
        vid = cv2.VideoCapture(args.file)
        run_processing(vid)
    elif args.stream is not None:
        vid = cv2.VideoCapture(args.stream)
        run_processing(vid)
    else:
        parser.print_help(sys.stdout)
