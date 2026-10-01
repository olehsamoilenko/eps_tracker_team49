"""Performance framework: FPS of the tracker's detectors on the Raspberry Pi, measured on a video, --frames (100)
frames after --warmup (10). No accuracy: see accuracy.py.

    python test_framework/performance.py                               # every models/*_ncnn detector
    python test_framework/performance.py --models yolo26 nanodet       # some of them
    python test_framework/performance.py --video input/my_video.mp4 --frames 300 --tracker bytetrack
    python test_framework/performance.py --video input/kyiv_drone.mp4 --frames 500 --chunk 50 --start-temp 60 \
        --max-temp 80                                                  # no fan: 10 chunks, each from 60 C
    python test_framework/performance.py --resume out/performance/20261001-153000

Each detector runs the pipeline (camera.open_source + pipeline.iter_frames in offline mode, as stream.py does for
`-s VIDEO --save`: every frame read, then detected) over the video in its own process, in one go: the Pi heats up as
in real use. Only after the video does the Pi cool down to the idle baseline (or --start-temp), so every detector
starts from the same state. With --start-temp, the timed frames also wait for it after the model load and warm-up.
With --chunk, the timed frames come in chunks spread evenly over the video, each one after a cooldown to
--start-temp, so a Pi without a fan never reaches its throttling point. Setup and details: TEST_FRAMEWORK.md
"""
import argparse
import os

from runner import (REPO, add_common_args, add_cooldown_args, cool_down, interrupted, is_done, log,  # first: puts
                    measure_baseline, open_run, run_detector, start_monitor)  # the repo root on sys.path
from camera import open_source
from report import performance_summary
from results import read_json, result_path, write_json
from tracker import TRACKERS


def count_frames(video, limit=None):
    """The video's frames (up to limit), read through the pipeline's own source (camera.open_source)."""
    cap, _, _ = open_source(video, 0, 0, 0)
    read = 0
    while (limit is None or read < limit) and cap.read()[0]:
        read += 1
    cap.release()
    return read


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=str(REPO / "input" / "fps_clip.mp4"), help="video to measure on")
    add_common_args(p, "performance")
    g = p.add_argument_group("measurement")
    g.add_argument("--frames", type=int, default=100, help="timed frames, right after the warm-up (default: 100)")
    g.add_argument("--warmup", type=int, default=10, help="untimed frames at the start of the video (default: 10)")
    g.add_argument("--tracker", choices=TRACKERS, default="none", help="tracker behind the detector (default: none)")
    g.add_argument("--chunk", type=int,
                   help="timed frames per chunk; the chunks spread evenly over the video, each one after a cooldown "
                        "to --start-temp (default: all --frames in one go)")
    add_cooldown_args(p)
    a = p.parse_args()
    if a.chunk and a.start_temp is None:
        p.error("--chunk needs --start-temp, the temperature every chunk starts at")
    if not a.resume:
        a.video = os.path.abspath(a.video)
        if not os.path.isfile(a.video):
            raise SystemExit(f"Video not found: {a.video} (see TEST_FRAMEWORK.md)")
        n = count_frames(a.video, None if a.chunk else a.warmup + a.frames)
        if n < a.warmup + a.frames:
            raise SystemExit(f"{a.video} is shorter than --warmup {a.warmup} + --frames {a.frames} frames")
        a.video_frames = n if a.chunk else None  # the worker spreads the chunks over the whole video
    return a


def main():
    a = parse_args()
    cfg = open_run(a, "performance", ("ncnn",))
    log.info(f"Video: {cfg['video']}")
    env, mon = start_monitor(cfg)
    try:
        base = measure_baseline(mon, cfg)
        todo = [n for n in cfg["models"] if not is_done(cfg, n)]
        if len(todo) < len(cfg["models"]):
            log.info(f"Already done, skipped: {', '.join(n for n in cfg['models'] if n not in todo)}")
        for i, name in enumerate(todo):
            if i == 0 and cfg.get("start_temp") is not None:  # the first detector also starts at --start-temp
                cool_down(mon, cfg, base, "baseline")
            run_detector(mon, cfg, name, ("fps",))
            performance_summary(cfg, env)  # out/summary.md gets every detector as soon as it is done
            if i < len(todo) - 1:  # cool down after a video, before the next detector, never during one
                cooldown = cool_down(mon, cfg, base, name)
                res = read_json(result_path(cfg, name))
                res["cooldown_after"] = cooldown
                write_json(result_path(cfg, name), res)
        mon.event("all detectors done")
    except KeyboardInterrupt:
        interrupted(mon, cfg)
    finally:
        mon.stop()
        performance_summary(cfg, env)


if __name__ == "__main__":
    main()
