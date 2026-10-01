"""NanoDet-Plus detector on ncnn: a model directory (model.ncnn.param/.bin, converted with pnnx) with metadata.yaml
giving imgsz [h, w], keep_ratio (letterbox, else stretch, as in training) and the class names."""
import os

import cv2
import ncnn
import numpy as np

from common import Detector, nms

MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
STRIDES = (8, 16, 32, 64)
REG_MAX = 7
THREADS = 4
IOU = 0.6  # NMS
MAX_DET = 500


class NanoDet(Detector):
    def __init__(self, path, conf):
        super().__init__(path, conf)
        self.keep_ratio = self.meta["keep_ratio"]
        self.nc = len(self.meta["names"])
        self.net = ncnn.Net()
        self.net.opt.num_threads = THREADS
        self.net.load_param(os.path.join(path, "model.ncnn.param"))
        self.net.load_model(os.path.join(path, "model.ncnn.bin"))
        self.priors = self._make_priors()
        out = self._infer(np.zeros((3, self.h, self.w), np.float32))  # warm-up
        if out.shape != (len(self.priors), self.nc + 4 * (REG_MAX + 1)):
            raise SystemExit(f"{path}: output {out.shape} does not match imgsz and names in metadata.yaml")

    def _make_priors(self):
        priors = []
        for s in STRIDES:
            fh, fw = int(np.ceil(self.h / s)), int(np.ceil(self.w / s))
            y, x = np.meshgrid(np.arange(fh) * s, np.arange(fw) * s, indexing="ij")
            priors.append(np.stack([x.ravel(), y.ravel(), np.full(x.size, s)], axis=1))
        return np.concatenate(priors).astype(np.float32)  # (N, 3): cx, cy, stride

    def _preprocess(self, img):
        # resize as in training: centered letterbox (keep_ratio) or stretch
        h0, w0 = img.shape[:2]
        if self.keep_ratio:
            r = min(self.w / w0, self.h / h0)
            M = np.array([[r, 0, (self.w - w0 * r) / 2], [0, r, (self.h - h0 * r) / 2]], dtype=np.float32)
        else:
            M = np.array([[self.w / w0, 0, 0], [0, self.h / h0, 0]], dtype=np.float32)
        x = cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)), M

    def _infer(self, x):
        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(x).clone())
        _, out = ex.extract("out0")
        return np.array(out)

    def predict(self, img):
        x, M = self._preprocess(img)
        out = self._infer(x)  # (N, num_classes + 4 * (REG_MAX + 1))
        scores = out[:, :self.nc]
        cls = scores.argmax(1)
        scores = scores[np.arange(len(cls)), cls]
        anchor = np.nonzero(scores > self.conf)[0]
        if not len(anchor):
            return np.zeros((0, 6), np.float32)
        scores, cls = scores[anchor], cls[anchor]
        reg, pri = out[anchor, self.nc:], self.priors[anchor]

        # distribution focal: softmax over REG_MAX+1 bins -> expected distance in strides
        reg = reg.reshape(-1, 4, REG_MAX + 1)
        reg = np.exp(reg - reg.max(-1, keepdims=True))
        dist = (reg / reg.sum(-1, keepdims=True)) @ np.arange(REG_MAX + 1, dtype=np.float32)
        dist *= pri[:, 2:3]
        boxes = np.stack([pri[:, 0] - dist[:, 0], pri[:, 1] - dist[:, 1],
                          pri[:, 0] + dist[:, 2], pri[:, 1] + dist[:, 3]], axis=1)

        # back to source frame coordinates
        rx, ry, dx, dy = M[0, 0], M[1, 1], M[0, 2], M[1, 2]
        boxes = (boxes - [dx, dy, dx, dy]) / [rx, ry, rx, ry]
        h0, w0 = img.shape[:2]
        boxes = boxes.clip(0, [w0, h0, w0, h0])

        idx = nms(boxes, scores, cls, IOU, MAX_DET)
        return np.c_[boxes[idx], scores[idx], cls[idx]].astype(np.float32)
