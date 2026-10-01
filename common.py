"""Shared building blocks: newest-value slot between threads, detector interface, NMS."""
import os
import threading

import cv2
import numpy as np
import yaml

from stats import Timing

# every model must be trained on these classes, in this order, so all detectors report the same ids
VISDRONE = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


class Latest:
    """Newest value from a producer thread; consumers wait for a newer one and skip whatever they missed."""

    def __init__(self, value=None):
        self.value, self.seq = value, 0
        self.cv = threading.Condition()

    def put(self, value):
        with self.cv:
            self.value, self.seq = value, self.seq + 1
            self.cv.notify_all()

    def wait_new(self, seq):
        with self.cv:
            self.cv.wait_for(lambda: self.seq != seq)
            return self.value, self.seq


class Detector:
    """Detector adapter interface over a model directory: model.ncnn.param/.bin and metadata.yaml
    (imgsz [h, w], names {class id: name}, which must be VISDRONE). A subclass loads the net and implements
    predict(BGR frame) -> float32 (N, 6) array of [x1, y1, x2, y2, conf, class id] in frame pixels."""

    def __init__(self, path, conf):
        with open(os.path.join(path, "metadata.yaml")) as f:
            self.meta = yaml.safe_load(f)
        if list(self.meta["names"].values()) != VISDRONE:
            raise SystemExit(f"{path}: model classes are not VisDrone ({', '.join(VISDRONE)})")
        self.h, self.w = self.meta["imgsz"]
        self.conf = conf
        self.names = dict(enumerate(VISDRONE))
        self.keep = None  # class ids to keep (--classes), None = all
        self.timing = Timing()

    def predict(self, frame):
        raise NotImplementedError

    def detect(self, frame):
        """predict() + class filter, timed."""
        with self.timing:
            dets = self.predict(frame)
            if self.keep is not None:
                dets = dets[np.isin(dets[:, 5], self.keep)]
        return dets


def nms(boxes, scores, cls, iou, max_det):
    """Per-class NMS over (N, 4) [x1, y1, x2, y2] boxes -> indices of the kept ones, best first."""
    # shift boxes of different classes apart so they don't suppress each other
    b = boxes + cls[:, None] * (boxes.max() + 1)
    idx = cv2.dnn.NMSBoxes(np.c_[b[:, :2], b[:, 2:] - b[:, :2]].tolist(), scores.tolist(), 0, iou)
    return np.array(idx, dtype=int).ravel()[:max_det]
