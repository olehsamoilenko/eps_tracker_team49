"""Yolo-FastestV2 detector on ncnn: a model directory (model.ncnn.param/.bin, converted with pnnx from the dog-qiuqiu
model code, with the output activations and both scales folded into one output) and metadata.yaml giving imgsz [h, w],
anchors (w, h in input pixels: 3 for stride 16, then 3 for stride 32) and the class names.
Pre/postprocessing follow upstream test.py: BGR stretched to imgsz, 0..1; score = objectness * class probability."""
import os

import cv2
import ncnn
import numpy as np

from common import Detector, nms

STRIDES = (16, 32)
NA = 3  # anchors per cell
THREADS = 4
IOU = 0.4  # NMS, as upstream test.py
MAX_DET = 300


class YoloFastest(Detector):
    def __init__(self, path, conf):
        super().__init__(path, conf)
        self.nc = len(self.meta["names"])
        self.net = ncnn.Net()
        self.net.opt.num_threads = THREADS
        self.net.load_param(os.path.join(path, "model.ncnn.param"))
        self.net.load_model(os.path.join(path, "model.ncnn.bin"))
        self.grid, self.anchors = self._make_grid(np.array(self.meta["anchors"], np.float32).reshape(-1, NA, 2))
        out = self._infer(np.zeros((3, self.h, self.w), np.float32))  # warm-up
        if out.shape != (len(self.grid), 5 * NA + self.nc):
            raise SystemExit(f"{path}: output {out.shape} does not match imgsz and names in metadata.yaml")

    def _make_grid(self, anchors):
        """-> (N, 3) cells [col, row, stride] and (N, NA, 2) their anchors, stride 16 then 32, row-major."""
        grid, anch = [], []
        for s, a in zip(STRIDES, anchors):
            fh, fw = self.h // s, self.w // s
            y, x = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")
            grid.append(np.stack([x.ravel(), y.ravel(), np.full(x.size, s)], axis=1))
            anch.append(np.broadcast_to(a, (x.size, NA, 2)))
        return np.concatenate(grid).astype(np.float32), np.concatenate(anch)

    def _infer(self, x):
        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(x).clone())
        _, out = ex.extract("out0")
        return np.array(out).T  # (N, 5 * NA + nc): NA x [x, y, w, h] sigmoid, NA objectness, class softmax

    def predict(self, img):
        h0, w0 = img.shape[:2]
        x = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        out = self._infer(np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32) / 255)
        cls_prob = out[:, 5 * NA:]
        cls = cls_prob.argmax(1)  # classes are shared by the cell's anchors
        scores = out[:, 4 * NA:5 * NA] * cls_prob[np.arange(len(cls)), cls][:, None]  # (N, NA)
        cell, a = np.nonzero(scores > self.conf)
        if not len(cell):
            return np.zeros((0, 6), np.float32)
        scores, cls = scores[cell, a], cls[cell]
        reg, g = out[cell, :4 * NA].reshape(-1, NA, 4)[np.arange(len(a)), a], self.grid[cell]
        xy = (reg[:, :2] * 2 - 0.5 + g[:, :2]) * g[:, 2:3]
        wh = (reg[:, 2:] * 2) ** 2 * self.anchors[cell, a]
        boxes = np.c_[xy - wh / 2, xy + wh / 2]
        idx = nms(boxes, scores, cls, IOU, MAX_DET)

        # back to source frame coordinates
        boxes = boxes[idx] * [w0 / self.w, h0 / self.h, w0 / self.w, h0 / self.h]
        boxes = boxes.clip(0, [w0, h0, w0, h0])
        return np.c_[boxes, scores[idx], cls[idx]].astype(np.float32)
