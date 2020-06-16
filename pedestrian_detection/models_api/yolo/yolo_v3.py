from pathlib import Path
import numpy as np
import cv2

from models_api.yolo.model_output import ModelOutput


class YoloV3:
    _NORM_FACTOR = 1.0 / 255.0
    _PERSON_CLASS_ID = 0
    _PROBABILITY_THRESHOLD = 0.3
    _SIZE = 320

    def __init__(self):
        self.weights = Path("./models/yolo_v3/yolov3.weights")
        self.cfg = Path("./models/yolo_v3/yolov3.cfg")
        try:
            self._net = cv2.dnn.readNet(str(self.weights.absolute()), str(self.cfg.absolute()))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception as e:
            print(f"Error: occurs : {e}")
            raise e
        self.output_layers = self._net.getUnconnectedOutLayersNames()

    def inference(self, frame):
        inference_outputs = list()
        self._net.setInput(cv2.dnn.blobFromImage(frame,
                                                 scalefactor=self._NORM_FACTOR,
                                                 size=(self._SIZE, self._SIZE),
                                                 mean=(0, 0, 0),
                                                 swapRB=True,
                                                 crop=False
                                                 ))
        bboxes = list()
        confs = list()
        class_ids = list()

        height, width, _ = frame.shape
        outputs = self._net.forward(self.output_layers)
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    bboxes.append((x, y, w, h))
                    confs.append(float(conf))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(bboxes, confs, 0.5, 0.4)
        for i in range(len(bboxes)):
            if i in indexes:
                x, y, w, h = bboxes[i]
                # inference_outputs.append(ModelOutput(x, y, w, h, confs[i], class_ids[i]))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return bboxes, confs, frame
