"""One stage of one detector, started by runner.py in a fresh process so the model's RAM is freed on exit.

    python test_framework/worker.py accuracy|eval|fps DETECTOR RUN_DIR/run_config.json

accuracy: slow, temperature-paced pass of the pipeline over the test images -> NAME_dets.npy   (accuracy.py)
eval:     COCO metrics from NAME_dets.npy, in its own process so no detector sits in RAM     (accuracy.py)
fps:      the pipeline over a short video in one go, timed                                  (performance.py)
Each stage adds its results to NAME.json; everything printed here ends up in benchmark.log.
"""
import os
import resource
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))  # the tracker's modules live in the repo root
from camera import ImageFile, open_source
from metrics import coco_metrics, latency_stats
from pipeline import iter_frames, load_detector
from results import PHASE, PROGRESS, dets_path, now, read_json, result_path, write_json
from stats import Stats, Timing
from telemetry import cpu_temp
from tracker import load_tracker


class Images:
    """Test images as a frame source with the read()/release() interface of camera.py's sources."""

    def __init__(self, paths):
        self.paths = iter(paths)

    def read(self):
        path = next(self.paths, None)
        return ImageFile(path).read() if path else (False, None)

    def release(self):
        pass


class RecordedTiming(Timing):
    """The pipeline's detector timer (det.timing wraps every detect() call) that also keeps each duration."""

    def __init__(self):
        super().__init__()
        self.samples = []

    def __exit__(self, *exc):
        self.samples.append((time.perf_counter() - self.t) * 1000)
        super().__exit__(*exc)


class Pacer:
    """Keeps the Pi cool during the accuracy pass: a fixed pause after every image, and a long one whenever the CPU
    reaches pause_temp, until it is back at resume_temp."""

    def __init__(self, cfg):
        self.pause, self.hot, self.cool = cfg["pause"], cfg["pause_temp"], cfg["resume_temp"]
        self.paused_s, self.pauses = 0.0, 0

    def __call__(self):
        time.sleep(self.pause)
        temp = cpu_temp()
        if temp is None or temp < self.hot:
            return
        phase("paused")
        print(f"CPU {temp} C >= {self.hot} C: pausing until {self.cool} C", flush=True)
        t0 = time.time()
        while temp is not None and temp > self.cool:
            time.sleep(1)
            temp = cpu_temp()
        self.paused_s += time.time() - t0
        self.pauses += 1
        print(f"CPU {temp} C: going on after {time.time() - t0:.0f} s", flush=True)
        phase("accuracy")


def pipeline_frames(cap, fps, is_file, det, tracker):
    """The pipeline as stream.py runs it for a video file with --save: pipeline.iter_frames in offline mode, every
    frame read and then detected, nothing skipped."""
    return iter_frames(cap, fps, is_file, det, tracker, single=False, offline=True)


def run(det, frames, after_frame=None, limit=None, total=None):
    """Consumes `limit` frames at most of the pipeline's (frame, boxes) iterator; a next run() on the same iterator
    continues where this one stopped. Returns per-frame boxes ((M, 7) [x1, y1, x2, y2, track_id, conf, cls]),
    per-frame det.detect() ms, the wall seconds and the last frame's "WxH".
    after_frame() runs between frames, outside the detector's timing. total: frames expected, for the progress bar."""
    stats, boxes, times, size = Stats(det), [], [], None
    t0 = time.perf_counter()
    for frame, b in frames:
        size = f"{frame.shape[1]}x{frame.shape[0]}"
        times.append(det.timing.samples[-1])
        boxes.append(b)
        stats.tick(b)
        stats.log()
        if total:
            print(f"{PROGRESS}{len(boxes)} {total}", flush=True)
        if limit and len(boxes) >= limit:
            break
        if after_frame:
            after_frame()
    return boxes, np.array(times), time.perf_counter() - t0, size


def load(name, cfg):
    """The detector as the pipeline loads it, with a timer that keeps every detect() duration.
    Returns (detector, its module)."""
    phase("load")
    det = load_detector(name, cfg["conf"], None)  # conf None: the pipeline's default for this model family
    det.timing = RecordedTiming()
    return det, sys.modules[type(det).__module__]


def phase(stage):
    print(PHASE + stage, flush=True)


def peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kB on Linux, bytes on macOS
    return round(peak / (2**20 if sys.platform == "darwin" else 2**10))


def model_info(det, module):
    return {"input": f"{det.w}x{det.h}", "threads": module.THREADS, "iou": module.IOU, "versions": versions()}


def versions():
    v = {}
    for p in ("ncnn", "opencv-python", "opencv-python-headless", "numpy", "pycocotools", "PyYAML"):
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return v


def update_result(cfg, name, **fields):
    res = read_json(result_path(cfg, name)) or {"name": name}
    res.update(fields)
    write_json(result_path(cfg, name), res)
    return res


def accuracy(name, cfg):
    images = read_json(cfg["gt"])["images"]
    paths = [os.path.join(cfg["data"], "images", im["file_name"]) for im in images]
    det, module = load(name, cfg)
    conf = det.conf  # the deployment conf, for P/R/F1
    module.MAX_DET = cfg["max_det"]  # read by predict() on every call; one cap for all models keeps mAP comparable
    det.conf = cfg["eval_conf"]
    pacer = Pacer(cfg)

    phase("accuracy")
    print(f"accuracy pass: {len(paths)} images at conf {cfg['eval_conf']}, {cfg['pause']} s pause after each, "
          f"pausing at {cfg['pause_temp']} C until {cfg['resume_temp']} C", flush=True)
    frames = pipeline_frames(Images(paths), 0, True, det, load_tracker("none"))
    boxes, _, wall, _ = run(det, frames, after_frame=pacer, total=len(paths))
    if len(boxes) != len(paths):  # an unreadable image ends the source early
        raise SystemExit(f"Cannot read {paths[len(boxes)]}")
    # pipeline boxes [x1, y1, x2, y2, track_id, conf, cls] -> [image_id, x1, y1, x2, y2, conf, cls]
    dets = np.concatenate([np.c_[np.full((len(b), 1), im["id"]), b[:, :4], b[:, 5:]]
                           for im, b in zip(images, boxes)]).astype(np.float32)
    np.save(dets_path(cfg, name), dets)
    print(f"accuracy pass done: {len(dets)} detections in {wall / 60:.1f} min, "
          f"{pacer.pauses} temperature pauses ({pacer.paused_s:.0f} s)", flush=True)
    update_result(cfg, name, **model_info(det, module), conf=conf, eval_conf=cfg["eval_conf"],
                  max_det=cfg["max_det"],
                  accuracy_pass={"images": len(paths), "wall_s": round(wall), "pauses": pacer.pauses,
                                 "paused_s": round(pacer.paused_s), "total_dets": len(dets),
                                 "max_dets_per_image": max(len(b) for b in boxes)},
                  peak_rss_mb=peak_rss_mb(), inferred=now())


def evaluate(name, cfg):
    phase("eval")
    t, res = time.time(), read_json(result_path(cfg, name))
    dets = np.load(dets_path(cfg, name))
    if not len(dets):
        raise SystemExit("The model produced no detections at all")
    print(f"COCO evaluation of {len(dets)} detections...", flush=True)
    acc = coco_metrics(cfg["gt"], dets, cfg["max_det"], res["conf"])
    update_result(cfg, name, accuracy=acc, eval_s=round(time.time() - t, 1))
    print(f"accuracy: mAP50 {acc['mAP50']}  AR {acc['AR']}  P/R/F1@{res['conf']} {acc['P']}/{acc['R']}/{acc['F1']}",
          flush=True)


def fps(name, cfg):
    det, module = load(name, cfg)

    phase("warmup")
    cap, video_fps, is_file = open_source(cfg["video"], 0, 0, 0)  # as stream.py opens -s VIDEO
    frames = pipeline_frames(cap, video_fps, is_file, det, load_tracker(cfg["tracker"]))
    warm = run(det, frames, limit=cfg["warmup"])[0] if cfg["warmup"] else []

    phase("video")  # the next frames of the same pass, no pauses: the Pi heats up as it would in use
    boxes, times, wall, resolution = run(det, frames, limit=cfg["frames"], total=cfg["frames"])
    frames.close()
    cap.release()
    if len(boxes) < cfg["frames"]:
        raise SystemExit(f"{cfg['video']} ended after {len(warm) + len(boxes)} frames; it needs --warmup "
                         f"{cfg['warmup']} + --frames {cfg['frames']}")
    speed = dict(latency_stats(times), pipeline_fps=round(len(boxes) / wall, 2),
                 dets_per_frame=round(float(np.mean([len(b) for b in boxes])), 1))
    print(f"{len(boxes)} frames: detector {speed['fps']} FPS, {speed['mean_ms']} ms/frame, "
          f"pipeline {speed['pipeline_fps']} FPS, peak RAM {peak_rss_mb()} MB", flush=True)
    update_result(cfg, name, **model_info(det, module), conf=det.conf, tracker=cfg["tracker"],
                  video={"path": cfg["video"], "resolution": resolution, "fps": round(video_fps, 2),
                         "warmup_frames": len(warm), "frames": len(boxes)},
                  speed=speed, peak_rss_mb=peak_rss_mb(), measured=now())


def main():
    stage, name, config = sys.argv[1:]
    {"accuracy": accuracy, "eval": evaluate, "fps": fps}[stage](name, read_json(config))


if __name__ == "__main__":
    main()
