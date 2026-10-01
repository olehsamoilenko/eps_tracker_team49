"""Object trackers, selected with --tracker.

To add one (BoT-SORT, ...): subclass Tracker in its own module, implement update(), add it to load_tracker().
"""
import numpy as np

TRACKERS = ["none", "bytetrack", "sort_fast"]


class Tracker:
    """Interface: update(dets, frame) -> float32 (M, 7) array of [x1, y1, x2, y2, track_id, conf, cls].
    dets is the detector output (N, 6) [x1, y1, x2, y2, conf, cls]; frame is the BGR frame they came from.
    Called once per detector run, from the detector thread."""

    def update(self, dets, frame):
        # no tracking: every detection passes through with track_id = -1
        return np.insert(dets, 4, -1, axis=1)


def load_tracker(name):
    """name: one of TRACKERS."""
    if name == "bytetrack":
        from byte_tracker import ByteTracker  # imported here: byte_tracker imports this module
        return ByteTracker()
    if name == "sort_fast":
        from sort_fast import SortFast
        return SortFast()
    return Tracker()
