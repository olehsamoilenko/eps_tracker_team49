"""SORT (Bewley et al., 2016) on ultralytics building blocks, selected with --tracker sort_fast.

Kalman filter (all tracks predicted in one batch) + IoU + Hungarian matching; no low-conf pass and no
re-identification, so it is cheaper than ByteTrack. Unmatched detections start new tracks right away.
Defaults are tuned on VisDrone-MOT val with YOLOv8n (for NanoDet det_thresh 0.5).
Always runs with camera motion compensation (gmc.py): after the Kalman predict, tracks are shifted by how the camera
moved since the previous update.
"""
import numpy as np
from ultralytics.trackers.byte_tracker import STrack
from ultralytics.trackers.utils import matching
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH

from gmc import GMC, warp_xyah
from tracker import Tracker


class SortFast(Tracker):
    def __init__(self, det_thresh=0.45, max_age=30, min_hits=1, iou_threshold=0.1):
        self.det_thresh, self.iou_threshold = det_thresh, iou_threshold
        self.max_age = max_age  # counted in update() calls: detector runs, not video frames
        self.min_hits = min_hits  # a track is shown after this many matches
        self.kf = KalmanFilterXYAH()
        self.gmc = GMC()
        self.tracks, self.frame_id = [], 0

    def update(self, dets, frame):
        self.frame_id += 1
        dets = dets[dets[:, 4] > self.det_thresh]
        # STrack takes [cx, cy, w, h, detection index]
        new = [STrack(np.r_[(d[:2] + d[2:4]) / 2, d[2:4] - d[:2], i], d[4], d[5]) for i, d in enumerate(dets)]
        STrack.multi_predict(self.tracks)
        warp_xyah(self.tracks, self.gmc.apply(frame))  # every update, also without tracks, to keep the frame
        cost = matching.iou_distance(self.tracks, new)
        matches, _, unmatched = matching.linear_assignment(cost, 1 - self.iou_threshold)
        for it, idet in matches:
            self.tracks[it].update(new[idet], self.frame_id)
        for idet in unmatched:
            new[idet].activate(self.kf, self.frame_id)
            self.tracks.append(new[idet])
        self.tracks = [t for t in self.tracks if self.frame_id - t.frame_id <= self.max_age]
        out = [np.r_[t.xyxy, t.track_id, t.score, t.cls] for t in self.tracks
               if t.frame_id == self.frame_id and t.tracklet_len + 1 >= self.min_hits]
        return np.array(out, dtype=np.float32).reshape(-1, 7)
