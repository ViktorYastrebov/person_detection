import numpy as np
import time
from .feature_extractor import Extractor
from .nn_matching import NearestNeighborDistanceMetric
from .preprocessing import non_max_suppression
from .detection import Detection
from .tracker import Tracker


class DeepSort(object):
    def __init__(self, model_path):
        self.min_confidence = 0.3
        self.nms_max_overlap = 1.0

        self.extractor = Extractor(model_path, use_cuda=True)

        max_cosine_distance = 0.2
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        # try:
        start = time.time()
        features = self._get_features(bbox_xywh, ori_img)
        end = time.time()
        exec_time = end - start
        print(f"get features time : {(1000 * exec_time)} ms")
        # except:
        #    print('a')
        detections = [Detection(bbox_xywh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._xywh_to_xyxy_centernet(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs

    def _xywh_to_xyxy(self, bbox_xywh):
        """
        Converts from (x1,x2 w,h) -> x1,y1,x2,y2)
        """
        x1, y1, w, h = bbox_xywh
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(int(x1 + w), self.width - 1)
        y2 = min(int(y1 + h), self.height - 1)
        return int(x1), int(y1), x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


if __name__ == '__main__':
    pass
