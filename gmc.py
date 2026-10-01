"""Lightweight global motion compensation (GMC): how the camera moved between two tracker updates.

BoT-SORT's CMC idea (corners + Lucas-Kanade flow + RANSAC similarity), made cheap for a Raspberry Pi: the frame is
downscaled to 320 px wide, ~200 spread corners are used and carried over between frames (searched again only when
fewer than half survive). ~1.7 ms/frame on a laptop vs 6-20 ms for ultralytics' GMC, with the same tracking quality.
"""
import cv2
import numpy as np

WIDTH, CORNERS = 320, 200
LK = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class GMC:
    def __init__(self):
        self.prev, self.pts = None, None

    def apply(self, frame):
        """frame: BGR. Returns a 2x3 affine mapping previous-frame pixels to this frame's (full resolution)."""
        H = np.eye(2, 3)
        k = max(frame.shape[1] / WIDTH, 1.0)
        # downscale first, then grey: cvtColor on the full frame costs more than the rest of GMC
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=1 / k, fy=1 / k), cv2.COLOR_BGR2GRAY)
        pts = np.zeros((0, 1, 2), np.float32)
        if self.prev is not None and len(self.pts) >= 8:
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, **LK)
            ok = ok.ravel().astype(bool)
            if ok.sum() >= 8:
                M, inl = cv2.estimateAffinePartial2D(self.pts[ok], nxt[ok], method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.0)
                if M is not None:
                    H = M * [[1, 1, k], [1, 1, k]]  # rotation and scale do not depend on pixel scale, the shift does
                    pts = nxt[ok][inl.ravel().astype(bool)]  # background, carried over to the next frame
        if len(pts) < CORNERS // 2:
            pts = cv2.goodFeaturesToTrack(gray, CORNERS, 0.01, 8, blockSize=3)
            pts = pts if pts is not None else np.zeros((0, 1, 2), np.float32)
        self.prev, self.pts = gray, pts
        return H


def warp_xyah(tracks, H):
    """Moves Kalman states [cx, cy, a, h, vx, vy, va, vh] of ultralytics STracks into the current frame: centre and
    its velocity through the affine, height and its velocity scaled, aspect a unchanged (ultralytics' multi_gmc
    assumes an XYWH state and would rotate (a, h) as a point)."""
    if not tracks:
        return
    T = np.eye(8)
    T[0:2, 0:2] = T[4:6, 4:6] = H[:, :2]
    T[3, 3] = T[7, 7] = np.sqrt(abs(np.linalg.det(H[:, :2])))
    mean = np.stack([t.mean for t in tracks]) @ T.T
    mean[:, :2] += H[:, 2]
    cov = T @ np.stack([t.covariance for t in tracks]) @ T.T
    for t, m, c in zip(tracks, mean, cov):
        t.mean, t.covariance = m, c
