# import the necessary packages

import os
import argparse
import sys
import time
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import imutils


def run_processing(video_stream: cv2.VideoCapture):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.destroyAllWindows()
    cv2.startWindowThread()


    # id = 0
    while True:
        return_value, frame = video_stream.read()
        if return_value:
            # detect people in the image
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = imutils.resize(frame, width=min(400, frame.shape[1]))
            start_time = time.time()

            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)
            # draw the original bounding boxes
            for (x, y, w, h) in rects:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

            end_time = time.time()
            exec_time = end_time - start_time
            info = "time: %.2f ms" % (1000 * exec_time)
            print(info)
            # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"file1_{id}.png", frame)
            # id += 1
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.destroyAllWindows()
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