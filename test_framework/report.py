"""summary.md and summary.csv of a run: accuracy.py -> accuracy and test conditions, performance.py -> FPS and
test conditions. Both are also written to benchmark.log."""
import csv
import logging
import os

from results import read_json, result_path
from telemetry import aggregate, telemetry_rows

log = logging.getLogger("benchmark")


def fmt(v, nd=None):
    if v is None or v == "":
        return "-"
    return f"{v:.{nd}f}" if nd is not None and isinstance(v, (int, float)) else str(v)


def table(rows, cols):
    """cols: (header, key, decimals or None)."""
    lines = ["| " + " | ".join(h for h, _, _ in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(k), nd) for _, k, nd in cols) + " |" for r in rows]
    return "\n".join(lines)


def pct(mb, total_mb):
    return round(100 * mb / total_mb, 1) if mb is not None and total_mb else None


def ram_text(mb, total_mb):
    return "-" if mb is None else f"{mb:.0f} MB" + (f" ({pct(mb, total_mb):.0f}%)" if total_mb else "")


def conditions(r, ram_mb):
    """Pi state during the detector's run (its worker processes), from the telemetry."""
    t = r.get("telemetry") or {}
    run = t.get("run", {})
    return {
        "status": r.get("status", "not run"), "run_min": round(run["seconds"] / 60, 1) if run else None,
        "start_temp_c": run.get("start_temp_c"), "end_temp_c": run.get("end_temp_c"),
        "max_temp_c": run.get("temp_c_max"), "min_mhz": run.get("arm_mhz_min"),
        "min_5v": run.get("ext5v_v_min"), "max_5v": run.get("ext5v_v_max"),
        "min_core_v": run.get("core_v_min"), "max_core_v": run.get("core_v_max"),
        "avg_ram_used_mb": run.get("mem_used_mb_mean"), "max_ram_used_mb": run.get("mem_used_mb_max"),
        "max_ram_used_pct": pct(run.get("mem_used_mb_max"), ram_mb),
        "avg_worker_mb": run.get("worker_rss_mb_mean"), "max_worker_mb": run.get("worker_rss_mb_max"),
        "flags": "|".join(run.get("flags", [])),
    }


CONDITION_COLS = [
    ("start °C", "start_temp_c", 1), ("end °C", "end_temp_c", 1), ("max °C", "max_temp_c", 1),
    ("min MHz", "min_mhz", 0), ("min 5V", "min_5v", 2), ("max 5V", "max_5v", 2), ("min core V", "min_core_v", 3),
    ("max core V", "max_core_v", 3), ("avg RAM used", "avg_ram_md", None), ("max RAM used", "max_ram_md", None),
    ("avg detector MB", "avg_worker_mb", 0), ("max detector MB", "max_worker_mb", 0), ("flags", "flags", None)]


def whole_run(cfg, env):
    """One line on the Pi's state over the whole session, from the first telemetry row to the last."""
    try:
        b = aggregate(telemetry_rows(cfg), cfg["interval"])
    except FileNotFoundError:
        return "-"

    def span(lo, hi, nd):
        return "-" if lo is None else f"{lo:.{nd}f}–{hi:.{nd}f} V"

    return (f"{fmt(b.get('start_temp_c'), 1)} → {fmt(b.get('end_temp_c'), 1)} °C "
            f"(max {fmt(b.get('temp_c_max'), 1)} °C), 5 V input {span(b.get('ext5v_v_min'), b.get('ext5v_v_max'), 2)}, "
            f"core {span(b.get('core_v_min'), b.get('core_v_max'), 3)}, RAM used avg "
            f"{ram_text(b.get('mem_used_mb_mean'), env['ram_mb'])}, max {ram_text(b.get('mem_used_mb_max'), env['ram_mb'])}, "
            f"flags: {', '.join(b.get('flags', [])) or 'none'}")


def write(cfg, env, rows, title, settings, sections):
    """summary.csv from rows, summary.md from the sections [(heading, text)], both logged."""
    with open(os.path.join(cfg["out"], "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, [k for k in rows[0] if not k.endswith("_md")])
        w.writeheader()
        w.writerows([{k: v for k, v in r.items() if not k.endswith("_md")} for r in rows])
    sections = sections + [("Whole run", f"Over the whole session (all detectors and cooldowns): "
                                         f"{whole_run(cfg, env)}")]
    text = f"# {title}: {env['device']}\n\n" + "".join(f"- {s}\n" for s in settings) \
        + f"- {env['os']}, kernel {env['kernel']}, CPU governor {env['cpu_governor']}, RAM {env['ram_mb']} MB\n" \
        + "".join(f"\n## {h}\n\n{body}\n" for h, body in sections)
    with open(os.path.join(cfg["out"], "summary.md"), "w") as f:
        f.write(text)
    log.info("Results\n\n" + "\n\n".join(body for _, body in sections) + "\n")
    log.info(f"Results saved: {os.path.join(cfg['out'], 'summary.md')} (and summary.csv, <detector>.json); "
             f"log: benchmark.log, telemetry: telemetry.csv")


def accuracy_summary(cfg, env):
    results = {name: read_json(result_path(cfg, name)) or {} for name in cfg["models"]}
    rows = []
    for name, r in results.items():
        a, p = r.get("accuracy") or {}, r.get("accuracy_pass") or {}
        row = {"model": name, "input": r.get("input"), "conf": r.get("conf"),
               **{k: a.get(k) for k in ("mAP50", "AR", "P", "R", "F1", "best_F1", "best_F1_conf")},
               **conditions(r, env["ram_mb"]), "pauses": p.get("pauses"), "paused_s": p.get("paused_s")}
        row["avg_ram_md"] = ram_text(row["avg_ram_used_mb"], env["ram_mb"])
        row["max_ram_md"] = ram_text(row["max_ram_used_mb"], env["ram_mb"])
        rows.append(row)
    accuracy = table(rows, [
        ("model", "model", None), ("input", "input", None), ("mAP50", "mAP50", 3),
        (f"AR@{cfg['max_det']}", "AR", 3), ("conf", "conf", 2), ("P", "P", 3), ("R", "R", 3), ("F1", "F1", 3),
        ("best F1", "best_F1", 3), ("at conf", "best_F1_conf", 2)])
    state = table(rows, [("model", "model", None), ("status", "status", None), ("run min", "run_min", 1),
                         ("hot pauses", "pauses", None), ("paused s", "paused_s", None)] + CONDITION_COLS)
    write(cfg, env, rows, "Detector accuracy", [
        f"Test set: `{cfg['data']}`, {cfg.get('n_images')} images, {cfg.get('n_boxes')} boxes"
        + (" (--limit subset)" if cfg["limit"] else ""),
        f"All at IoU 0.5 (pycocotools): mAP50 and AR with detections of score >= {cfg['eval_conf']}, up to "
        f"{cfg['max_det']} per image; P/R/F1 at each detector's `conf`",
        f"Pacing: {cfg['pause']} s rest after every image; paused at {cfg['pause_temp']} °C until "
        f"{cfg['resume_temp']} °C"], [
        ("Accuracy", accuracy),
        ("Conditions", "Per detector, over its accuracy pass and COCO evaluation. RAM used: the whole system; "
                       "detector MB: its worker process.\n\n" + state)])


def performance_summary(cfg, env):
    rows = []
    for name in cfg["models"]:
        r = read_json(result_path(cfg, name)) or {}
        s, v = r.get("speed") or {}, r.get("video") or {}
        video_flags = (r.get("telemetry") or {}).get("video", {}).get("flags", [])
        row = {"model": name, "input": r.get("input"), "conf": r.get("conf"), "warmup_frames": v.get("warmup_frames"),
               "frames": v.get("frames"), "fps": s.get("fps"), "mean_ms": s.get("mean_ms"),
               "pipeline_fps": s.get("pipeline_fps"), "peak_rss_mb": r.get("peak_rss_mb"),
               **conditions(r, env["ram_mb"]), "video_flags": "|".join(video_flags),
               "cooldown_after_s": (r.get("cooldown_after") or {}).get("seconds")}
        # FPS measured while the Pi was throttling or under-volted is not clean
        row["fps_md"] = fmt(row["fps"], 2) + (" (!)" if video_flags else "")
        row["avg_ram_md"] = ram_text(row["avg_ram_used_mb"], env["ram_mb"])
        row["max_ram_md"] = ram_text(row["max_ram_used_mb"], env["ram_mb"])
        rows.append(row)
    speed = table(rows, [
        ("model", "model", None), ("input", "input", None), ("warm-up frames", "warmup_frames", None),
        ("frames", "frames", None), ("detector FPS", "fps_md", None), ("mean ms", "mean_ms", 1),
        ("pipeline FPS", "pipeline_fps", 2), ("peak RAM MB", "peak_rss_mb", 0)])
    state = table(rows, [("model", "model", None), ("status", "status", None), ("run min", "run_min", 1)]
                  + CONDITION_COLS + [("cooldown after s", "cooldown_after_s", 0)])
    video = next((r["video"] for r in (read_json(result_path(cfg, n)) or {} for n in cfg["models"])
                  if r.get("video")), {})
    write(cfg, env, rows, "Detector FPS", [
        f"Video: `{cfg['video']}`, {video.get('resolution', '?')} at {video.get('fps', '?')} FPS; "
        f"{cfg['warmup']} warm-up frames, then {cfg['frames']} timed frames in one go",
        f"Tracker: {cfg['tracker']}; detection confidence: each detector's `conf`",
        f"Cooldown to {cfg['start_temp']} °C before every detector, never during a video"
        if cfg.get("start_temp") is not None else
        f"Cooldown to the idle baseline (+{cfg['cool_delta']} °C) after each video, never during one"], [
        ("Speed", "Warm-up frames: untimed, at the start of the video. Detector FPS and mean ms: `det.detect()` "
                  "(preprocess + inference + postprocess) over the timed frames, timed like the pipeline's "
                  "`det.timing`. Pipeline FPS: frames per second of `pipeline.iter_frames()` (offline), including "
                  "video decoding and the tracker. (!): throttling or under-voltage during the video, so that FPS "
                  "is not clean.\n\n" + speed),
        ("Conditions", "Per detector, from its worker's start (model load, warm-up) to the end of the video. "
                       "RAM used: the whole system; detector MB: its worker process.\n\n" + state)])
