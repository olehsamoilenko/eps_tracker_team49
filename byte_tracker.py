"""ByteTrack (Zhang et al., 2022) from ultralytics, selected with --tracker bytetrack.

High-conf detections are matched to all tracks, then low-conf ones (low_thresh..high_thresh) to the tracks
still unmatched; unmatched high ones with conf >= new_thresh start new tracks.
Low detections only reach the tracker if the detector's --conf is at or below low_thresh.
Defaults are tuned on VisDrone-MOT val with YOLOv8n (ultralytics' own: 0.25 / 0.1 / 0.25 / 0.8).
Always runs with camera motion compensation (gmc.py): after the Kalman predict, tracks are shifted by how the camera
moved since the previous update. On VisDrone-MOT val with YOLOv8n: IDF1 47.7 -> 51.3, ID switches 197 -> 77.
"""
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from gmc import GMC, warp_xyah
from tracker import Tracker


class BYTETrackerGMC(BYTETracker):
    def __init__(self, args):
        if not hasattr(BYTETracker, "_pre_first_associate"):  # older ultralytics would silently skip GMC
            raise SystemExit("ByteTrack with GMC needs ultralytics with BYTETracker._pre_first_associate (8.4+)")
        super().__init__(args)
        self.cam = GMC()  # not self.gmc: ultralytics would call that one with its own (XYWH) warp

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high):
        # ultralytics hook: runs after the Kalman predict, before the first matching stage
        H = self.cam.apply(img)  # called on every update, also without tracks, to keep the previous frame
        warp_xyah(strack_pool + unconfirmed, H)


class ByteTracker(Tracker):
    def __init__(self, high_thresh=0.35, low_thresh=0.1, new_thresh=0.65, match_thresh=0.9, track_buffer=30):
        # track_buffer is counted in update() calls: detector runs, not video frames
        self.bt = BYTETrackerGMC(SimpleNamespace(
            track_high_thresh=high_thresh, track_low_thresh=low_thresh, new_track_thresh=new_thresh,
            match_thresh=match_thresh, track_buffer=track_buffer, fuse_score=True))

    def update(self, dets, frame):
        out = self.bt.update(Boxes(dets, frame.shape[:2]), frame)  # (M, 8): ..., detection index
        return np.asarray(out, dtype=np.float32).reshape(-1, 8)[:, :7]
