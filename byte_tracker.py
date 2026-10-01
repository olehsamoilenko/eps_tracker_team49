"""ByteTrack (Zhang et al., 2022), selected with --tracker bytetrack. Stub: detections pass through untracked.

TODO, per update() after the Kalman filter predicts every track:
  1. split detections by conf: high >= high_thresh, low in [low_thresh, high_thresh)
  2. match tracked + lost tracks to high detections by IoU (Hungarian, cost 1 - IoU <= match_thresh)
  3. match the still-unmatched tracked tracks to low detections (cost <= 0.5); the rest become lost
  4. unmatched high detections with conf >= new_thresh start new tracks
  5. drop lost tracks older than track_buffer updates
Low detections only reach the tracker if the detector's --conf is at or below low_thresh.
"""
from tracker import Tracker


class ByteTracker(Tracker):
    def __init__(self, high_thresh=0.25, low_thresh=0.1, new_thresh=0.25, match_thresh=0.8, track_buffer=30):
        self.high_thresh, self.low_thresh, self.new_thresh = high_thresh, low_thresh, new_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer  # counted in update() calls: detector runs, not video frames
        self.tracks, self.next_id = [], 0
        print("ByteTrack is a stub: boxes get no track IDs yet", flush=True)

    def update(self, dets, frame):
        # TODO: steps 1-5 above
        return super().update(dets, frame)
