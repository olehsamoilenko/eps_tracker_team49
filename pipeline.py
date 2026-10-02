"""Detection pipeline: source -> detector (+ tracker) -> (frame, boxes) pairs.

boxes is a float32 (M, 7) array of [x1, y1, x2, y2, track_id, conf, cls], or None without a detector.
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from common import Latest

MODELS = Path(__file__).resolve().parent / "models"
# each models/NAME_ncnn directory is a detector: nanodet* are NanoDet-Plus, yolo_fastest* Yolo-FastestV2, the rest
# ultralytics YOLO exports (exported with ultralytics 8.4; an older yolo8 export segfaulted in ncnn 1.0.20260526)
DETECTORS = sorted(p.name.removesuffix("_ncnn") for p in MODELS.glob("*_ncnn"))
DEFAULT_CONF = {"yolo": 0.25, "nanodet": 0.35, "yolo_fastest": 0.3}  # yolo_fastest: upstream test.py
NO_BOXES = np.zeros((0, 7), np.float32)


def load_detector(name, conf, classes):
    """name: one of DETECTORS or None; conf: None for the default; classes: 'pedestrian,car' or None."""
    if name is None:
        return None
    path = str(MODELS / f"{name}_ncnn")
    family = next((f for f in ("nanodet", "yolo_fastest") if name.startswith(f)), "yolo")
    conf = DEFAULT_CONF[family] if conf is None else conf
    if family == "yolo":
        from yolo_detector import Yolo  # ncnn is imported only when a detector is used
        det = Yolo(path, conf)
    elif family == "yolo_fastest":
        from yolo_fastest_detector import YoloFastest
        det = YoloFastest(path, conf)
    else:
        from nanodet_detector import NanoDet
        det = NanoDet(path, conf)
    if classes:
        wanted = classes.split(",")
        unknown = set(wanted) - set(det.names.values())
        if unknown:
            raise SystemExit(f"Unknown classes {sorted(unknown)}; the model has: {', '.join(det.names.values())}")
        det.keep = [c for c, n in det.names.items() if n in wanted]
    return det


def iter_frames(cap, fps, is_file, det, tracker, single, offline, max_det_fps=0):
    """single: one frame; offline: every frame of a file is detected, until its end; otherwise live_frames(),
    where max_det_fps > 0 caps the detector's runs per second."""
    if single:
        return one_frame(cap, fps, is_file, det, tracker)
    if offline:
        return file_frames(cap, det, tracker)
    return live_frames(cap, fps, is_file, det, tracker, max_det_fps)


def analyze(frame, det, tracker):
    return tracker.update(det.detect(frame), frame) if det else None


def one_frame(cap, fps, is_file, det, tracker):
    """A single frame; a camera first runs for ~1 s so auto-exposure settles."""
    for _ in range(0 if is_file else int(fps)):
        cap.read()
    ok, frame = cap.read()
    if ok:
        yield frame, analyze(frame, det, tracker)


def file_frames(cap, det, tracker):
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame, analyze(frame, det, tracker)


def live_frames(cap, fps, is_file, det, tracker, max_det_fps=0):
    """Frames at source speed; detection runs in its own thread and its latest result goes with each new frame."""
    frames, boxes, stop = Latest(), Latest(NO_BOXES), threading.Event()
    workers = [threading.Thread(target=capture_loop, args=(cap, fps, is_file, frames, stop), daemon=True)]
    if det:
        workers.append(threading.Thread(target=detect_loop, args=(det, tracker, frames, boxes, stop, max_det_fps),
                                        daemon=True))
    for w in workers:
        w.start()
    try:
        seq = 0
        while True:
            frame, seq = frames.wait_new(seq)
            if frame is None:
                return
            # copy: the detector thread may still be reading this frame while we draw on it
            yield (frame.copy(), boxes.value) if det else (frame, None)
    finally:
        # let workers finish before the source is released and Python exits
        # (a native library can crash if the interpreter shuts down while a thread is inside it)
        stop.set()
        frames.put(None)  # wakes the detector
        for w in workers:
            w.join(timeout=5)


def capture_loop(cap, fps, is_file, frames, stop):
    """Reads the source into `frames`; a video file is paced at its native FPS and looped, like a live camera."""
    t0, n = time.perf_counter(), 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok and is_file and n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            t0, n = time.perf_counter(), 0
            continue
        if not ok:
            break
        if is_file:
            n += 1
            time.sleep(max(0.0, t0 + n / fps - time.perf_counter()))
        frames.put(frame)
    frames.put(None)  # end of stream


def detect_loop(det, tracker, frames, boxes, stop, max_det_fps=0):
    seq = 0
    while not stop.is_set():
        t = time.perf_counter()
        frame, seq = frames.wait_new(seq)
        if frame is None:
            return
        boxes.put(analyze(frame, det, tracker))
        if max_det_fps:
            stop.wait(max(0.0, t + 1 / max_det_fps - time.perf_counter()))  # idle until the next run is due
