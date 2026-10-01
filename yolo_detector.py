"""YOLO (v8 / 11 / 26) detector on ncnn: a model directory exported by ultralytics (model.ncnn.param/.bin +
metadata.yaml). Pre/postprocessing follow ultralytics predict(), so results match it without needing torch.
YOLO26 must be exported with end2end: false (its default for ncnn), which gives the same output as v8."""
import os

import cv2
import ncnn
import numpy as np

from common import Detector, nms

THREADS = 4
IOU = 0.7  # NMS
MAX_DET = 300
PAD = 114  # letterbox border gray


class Yolo(Detector):
    def __init__(self, path, conf):
        super().__init__(path, conf)
        self.net = ncnn.Net()
        self.net.opt.num_threads = THREADS
        self.net.load_param(os.path.join(path, "model.ncnn.param"))
        self.net.load_model(os.path.join(path, "model.ncnn.bin"))
        self._infer(np.zeros((3, self.h, self.w), np.float32))  # warm-up

    def _preprocess(self, img):
        # centered letterbox with a gray border, RGB in 0..1
        h0, w0 = img.shape[:2]
        r = min(self.w / w0, self.h / h0)
        w1, h1 = round(w0 * r), round(h0 * r)
        if (w1, h1) != (w0, h0):
            img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_LINEAR)
        top, left = (self.h - h1) // 2, (self.w - w1) // 2
        x = cv2.copyMakeBorder(img, top, self.h - h1 - top, left, self.w - w1 - left,
                               cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
        return cv2.dnn.blobFromImage(x, 1 / 255, swapRB=True)[0], r, left, top

    def _infer(self, x):
        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(x).clone())
        _, out = ex.extract("out0")
        return np.array(out)

    def predict(self, img):
        x, r, dx, dy = self._preprocess(img)
        out = self._infer(x).T  # (N, 4 + num_classes): cx, cy, w, h in input pixels, then class scores
        scores = out[:, 4:]
        cls = scores.argmax(1)
        scores = scores[np.arange(len(cls)), cls]
        anchor = np.nonzero(scores > self.conf)[0]
        if not len(anchor):
            return np.zeros((0, 6), np.float32)
        scores, cls = scores[anchor], cls[anchor]
        xy, wh = out[anchor, :2], out[anchor, 2:4]
        boxes = np.c_[xy - wh / 2, xy + wh / 2]
        idx = nms(boxes, scores, cls, IOU, MAX_DET)

        # back to source frame coordinates
        boxes = (boxes[idx] - [dx, dy, dx, dy]) / r
        h0, w0 = img.shape[:2]
        boxes = boxes.clip(0, [w0, h0, w0, h0])
        return np.c_[boxes, scores[idx], cls[idx]].astype(np.float32)
