import sys
import time
import argparse
import cv2
from pathlib import Path
from dataclasses import dataclass


class Inference:
    def __init__(self, pd: Path, pdtxt: Path, threshold: float):
        self.tensorflow_net = cv2.dnn.readNetFromTensorflow(str(pd), str(pdtxt))
        self.tensorflow_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.tensorflow_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self._threshold = threshold

    def process(self, img):
        rows, cols, channels = img.shape
        # Use the given image as input, which needs to be blob(s).
        self.tensorflow_net.setInput(cv2.dnn.blobFromImage(img, size=(600, 600), swapRB=True, crop=False))
        # Runs a forward pass to compute the net output
        network_output = self.tensorflow_net.forward()
        print(f"shapes: {network_output.shape}")

        # Loop on the outputs
        for detection in network_output[0, 0]:
            score = float(detection[2])
            if score > self._threshold:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                # draw a blue rectangle around detected objects
                cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        return img


def run_processing(video_stream: cv2.VideoCapture, pd, pbtxt):
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    inference = Inference(pd, pbtxt, 0.6)
    while True:
        return_value, frame = video_stream.read()
        if return_value:
            start_time = time.time()
            out_frame = inference.process(frame)
            end_time = time.time()
            print(f"time: {1000 * (end_time - start_time)} ms for frame")
            result = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            cv2.destroyAllWindows()
            break


@dataclass
class ModelConfig:
    pb: Path
    pbtxt: Path


class ModelSelector:
    models = {
        'ssdlite_mobilenet_v2_coco': 'ssdlite_mobilenet_v2_coco_2018_05_09',
        'ssd_mobilenet_v2_coco': 'ssd_mobilenet_v2_coco_2018_03_29',
        'fast_rcnn_v2_coco':  'faster_rcnn_inception_v2_coco_2018_01_28'
    }
    base_dir = Path("./model")

    def get(self, name: str):
        ret = self.models.get(name)
        if ret is None:
            raise ValueError("Unknown model has been selected")

        mc = ModelConfig(Path(self.base_dir / ret / "frozen_inference_graph.pb"),
                         Path(self.base_dir / ret / "frozen_inference_graph.pbtxt"))
        return mc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--file', type=str, help='Specify video file for processing')
    parser.add_argument('--stream', type=int, help='Specify video stream for processing')
    parser.add_argument('--model_name', type=str, help="Specify one of the model name:"
                                                       "ssdlite_mobilenet_v2_coco,"
                                                       "ssd_mobilenet_v2_coco,"
                                                       "fast_rcnn_v2_coco")

    model_selector = ModelSelector()

    args = parser.parse_args()
    model_name = 'ssdlite_mobilenet_v2_coco'
    if args.model_name is not None:
        model_name = args.model_name

    model_config = model_selector.get(model_name)

    if args.file is not None:
        vid = cv2.VideoCapture(args.file)
        print(f"Selected model name: {model_name}")
        run_processing(vid, model_config.pb, model_config.pbtxt)
    elif args.stream is not None:
        vid = cv2.VideoCapture(args.stream)
        print(f"Selected model name: {model_name}")
        run_processing(vid, model_config.pb, model_config.pbtxt)
    else:
        parser.print_help(sys.stdout)
