"""Accuracy framework: COCO metrics of the tracker's detectors on VisDrone2019-DET test-dev (never used in
training; by default its 500-image subset with the same class ratio, made by subset.py), measured on the Raspberry
Pi slowly enough that it never overheats. No FPS: see performance.py.

    python test_framework/accuracy.py                                  # every models/*_ncnn detector
    python test_framework/accuracy.py --models yolo26 nanodet          # some of them
    python test_framework/accuracy.py --limit 50 --out out/accuracy/smoke
    python test_framework/accuracy.py --resume out/accuracy/20261001-153000   # continue after a crash

Each detector runs the pipeline over the test images in its own process, with a pause after every image and a
longer one whenever the CPU reaches --pause-temp, until it is back at --resume-temp. The COCO evaluation then runs
in another process. Average and max CPU temperature and RAM of each detector go to the log and to out/summary.md.
Setup and details: TEST_FRAMEWORK.md
"""
import argparse
import os

from runner import REPO, add_common_args, interrupted, log, open_run, run_detector, start_monitor  # first: puts the repo root on sys.path
from dataset import build_gt
from report import accuracy_summary, ram_text
from results import write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(REPO / "input" / "VisDrone2019-DET-test-dev-500"),
                   help="VisDrone folder with images/ and annotations/ (default: the 500-image subset)")
    p.add_argument("--limit", type=int, default=0, help="use N images spread over the set (default: all)")
    add_common_args(p, "accuracy")
    g = p.add_argument_group("measurement")
    g.add_argument("--eval-conf", type=float, default=0.001, help="confidence for mAP (default: 0.001)")
    g.add_argument("--max-det", type=int, default=500, help="max detections per image, all models (default: 500)")
    g = p.add_argument_group("pacing")
    g.add_argument("--pause", type=float, default=0.5, help="seconds of rest after every image (default: 0.5)")
    g.add_argument("--pause-temp", type=float, default=70,
                   help="pause the pass when the CPU reaches this, C (default: 70)")
    g.add_argument("--resume-temp", type=float, default=60, help="... until it is back at this, C (default: 60)")
    a = p.parse_args()
    if not a.resume:
        a.data = os.path.abspath(a.data)
        if not all(os.path.isdir(os.path.join(a.data, d)) for d in ("images", "annotations")):
            raise SystemExit(f"{a.data} must contain images/ and annotations/ (see TEST_FRAMEWORK.md; the "
                             f"500-image subset is made by test_framework/subset.py)")
    return a


def log_conditions(res, ram_mb):
    run = (res.get("telemetry") or {}).get("run")
    if run and run.get("temp_c_mean") is not None:
        log.info(f"{res['name']}: CPU avg {run['temp_c_mean']:.1f} C, max {run['temp_c_max']:.1f} C; RAM used avg "
                 f"{ram_text(run.get('mem_used_mb_mean'), ram_mb)}, max {ram_text(run.get('mem_used_mb_max'), ram_mb)}; "
                 f"detector process avg {run.get('worker_rss_mb_mean')} MB, max {run.get('worker_rss_mb_max')} MB")


def main():
    a = parse_args()
    cfg = open_run(a, "accuracy", ("ncnn", "pycocotools", "tqdm"))
    cfg["gt"] = os.path.join(cfg["out"], "gt.json")
    if not os.path.exists(cfg["gt"]):
        cfg["n_images"], cfg["n_boxes"] = build_gt(cfg["data"], cfg["limit"], cfg["gt"])
    log.info(f"Test set: {cfg['data']}, {cfg['n_images']} images, {cfg['n_boxes']} boxes")
    write_json(cfg["config_path"], cfg)
    env, mon = start_monitor(cfg)
    try:
        for name in cfg["models"]:
            res = run_detector(mon, cfg, name, ("accuracy", "eval"))
            log_conditions(res, env["ram_mb"])
            accuracy_summary(cfg, env)  # out/summary.md gets every detector as soon as it is done
        mon.event("all detectors done")
    except KeyboardInterrupt:
        interrupted(mon, cfg)
    finally:
        mon.stop()
        accuracy_summary(cfg, env)


if __name__ == "__main__":
    main()
