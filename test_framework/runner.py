"""Shared orchestration of the two frameworks (accuracy.py, performance.py): run folder, settings and --resume,
logging, telemetry monitor with the guard, idle baseline, cooldown, and one worker process per detector stage
(worker.py), so a detector's RAM goes back to the OS when its process exits."""
import contextlib
import importlib.util
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(REPO))  # the tracker's modules live in the repo root
from pipeline import DETECTORS
from results import PHASE, PROGRESS, now, read_json, result_path, setup_logging, write_json
from telemetry import IDLE_CPU_PCT, VCGENCMD, Monitor, environment, flag_names, phase_stats, stop_process

log = logging.getLogger("benchmark")
WORKER = Path(__file__).resolve().parent / "worker.py"


def add_common_args(p, kind):
    p.add_argument("--models", nargs="+", choices=DETECTORS,
                   help="detectors from models/, in this order (default: all); with --resume, detectors to add")
    p.add_argument("--out", help=f"results folder (default: out/{kind}/<date-time>)")
    p.add_argument("--resume", metavar="DIR", help="continue an interrupted run with its original settings")
    p.add_argument("--conf", type=float, help="detection confidence (default: the pipeline's, yolo* 0.25, "
                                              "nanodet* 0.35)")
    g = p.add_argument_group("telemetry and guard")
    g.add_argument("--interval", type=float, default=1.0, help="telemetry period, seconds (default: 1)")
    g.add_argument("--max-temp", type=float, default=82,
                   help="the guard stops a detector at this CPU temperature, C (default: 82)")
    g.add_argument("--min-free-mb", type=float, default=150,
                   help="the guard stops a detector when free RAM drops below this (default: 150)")
    g.add_argument("--no-undervolt-guard", dest="undervolt_guard", action="store_false",
                   help="keep running under under-voltage")


def add_cooldown_args(p):
    g = p.add_argument_group("cooldown")
    g.add_argument("--baseline-s", type=float, default=30, help="idle seconds measured at start (default: 30)")
    g.add_argument("--cool-delta", type=float, default=3.0,
                   help="cooled down at <= baseline temperature + this, C (default: 3)")
    g.add_argument("--start-temp", type=float,
                   help="cool down to this CPU temperature, C, before every detector, the first one included "
                        "(default: baseline + --cool-delta, after each video only)")
    g.add_argument("--cooldown-min", type=float, default=30, help="seconds (default: 30)")
    g.add_argument("--cooldown-max", type=float, default=600,
                   help="seconds, then go on anyway with a warning (default: 600)")
    g.add_argument("--ram-tolerance-mb", type=float, default=150,
                   help="free RAM must be back within this of the baseline (default: 150)")
    g.add_argument("--drop-caches", action="store_true",
                   help="drop the page cache after each detector (needs passwordless sudo)")


def open_run(a, kind, packages):
    """A new run folder out/KIND/<date-time> with the command-line settings, or the saved one with --resume.
    Starts logging to its benchmark.log. Returns the settings."""
    missing = [m for m in packages if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"Missing packages, install them into the venv: pip install {' '.join(missing)}")
    added = []
    if a.resume:
        out = os.path.abspath(a.resume)
        cfg = read_json(os.path.join(out, "run_config.json"))
        if not cfg:
            raise SystemExit(f"No run_config.json in {out}")
        added = [m for m in dict.fromkeys(a.models or []) if m not in cfg["models"]]
        cfg["models"] += added  # same settings, same summary
    else:
        out = os.path.abspath(a.out or REPO / "out" / kind / time.strftime("%Y%m%d-%H%M%S"))
        if os.path.exists(os.path.join(out, "run_config.json")):
            raise SystemExit(f"{out} already holds a run: use --resume {out} or another --out")
        os.makedirs(out, exist_ok=True)
        cfg = {k: v for k, v in vars(a).items() if k not in ("resume", "out")}
        cfg.update(kind=kind, models=list(dict.fromkeys(a.models or DETECTORS)), created=now())
    cfg.update(out=out, config_path=os.path.join(out, "run_config.json"))
    setup_logging(out)
    log.info(f"===== {kind}: {'resuming' if a.resume else 'new'} run {out}"
             + (" (with its original settings, other options are ignored)" if a.resume else "")
             + (f"; added detectors: {', '.join(added)}" if added else ""))
    return cfg


def start_monitor(cfg):
    """Saves the settings and the Pi's environment, starts the telemetry monitor. Returns (environment, monitor)."""
    write_json(cfg["config_path"], cfg)
    env = environment()
    write_json(os.path.join(cfg["out"], "environment.json"), env)
    log.info(f"{env['device']} | {env['os']} | governor {env['cpu_governor']} | RAM {env['ram_mb']} MB")
    log.info(f"Detectors: {', '.join(cfg['models'])}")
    mon = Monitor(cfg)
    mon.start()
    if not VCGENCMD:
        mon.event("vcgencmd not found: no voltage, clock or throttling telemetry (not a Raspberry Pi?)", alert=True)
    if env["cpu_governor"] not in (None, "performance"):
        mon.event(f"CPU governor is '{env['cpu_governor']}'; 'performance' gives steadier FPS "
                  "(see TEST_FRAMEWORK.md)", alert=True)
    if env["throttled_since_boot"]:
        mon.event(f"since boot the Pi already had: {env['throttled_since_boot']} (weak power supply or cooling?)",
                  alert=True)
    return env, mon


def interrupted(mon, cfg):
    mon.event(f"interrupted; continue with: python test_framework/{os.path.basename(sys.argv[0])} "
              f"--resume {cfg['out']}", alert=True)


def measure_baseline(mon, cfg):
    mon.phase = "baseline"
    log.info(f"Measuring the idle baseline for {cfg['baseline_s']:.0f} s...")
    rows = []
    for _ in range(max(1, round(cfg["baseline_s"] / cfg["interval"]))):
        time.sleep(cfg["interval"])
        rows.append(mon.last)

    def avg(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return round(sum(v) / len(v), 1) if v else None

    base = {k: avg(k) for k in ("temp_c", "arm_mhz", "cpu_pct", "mem_avail_mb")}
    mon.event(f"idle baseline: {base}")
    if (base["cpu_pct"] or 0) > IDLE_CPU_PCT:
        mon.event(f"the Pi is not idle ({base['cpu_pct']}% CPU): close other programs for clean FPS", alert=True)
    mon.phase = "idle"
    return base


def cool_down(mon, cfg, base, after):
    """Waits until temperature, free RAM, CPU load and throttling flags are back to the baseline."""
    mon.phase = f"cooldown after {after}"
    if cfg["drop_caches"]:
        r = subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                           capture_output=True, text=True)
        if r.returncode:
            mon.event(f"--drop-caches failed (needs passwordless sudo): {r.stderr.strip()}", alert=True)
    target = cfg.get("start_temp")  # .get: runs saved before --start-temp existed
    if target is None and base["temp_c"] is not None:
        target = base["temp_c"] + cfg["cool_delta"]
    t0, temps = time.time(), deque()
    while True:
        time.sleep(cfg["interval"])
        s, waited = mon.last, time.time() - t0
        temp = s.get("temp_c")
        temps.append((waited, temp))
        while temps[0][0] < waited - 60:
            temps.popleft()
        busy = []
        if target is not None and temp is not None and temp > target:
            # a plateau above the target (warm room, passive cooling) also counts as cooled down
            plateau = waited >= 60 and temps[0][1] is not None and temps[0][1] - temp < 0.5
            if not plateau:
                busy.append(f"CPU {temp} C > {target:.1f} C")
        if base["mem_avail_mb"] and s.get("mem_avail_mb") is not None \
                and s["mem_avail_mb"] < base["mem_avail_mb"] - cfg["ram_tolerance_mb"]:
            busy.append(f"free RAM {s['mem_avail_mb']} MB < baseline {base['mem_avail_mb']:.0f} MB")
        if (s.get("cpu_pct") or 0) > IDLE_CPU_PCT:
            busy.append(f"CPU load {s['cpu_pct']}%")
        if (s.get("throttled") or 0) & 0xF:
            busy.append(f"flags {flag_names(s['throttled'])}")
        if waited >= cfg["cooldown_min"] and not busy:
            break
        if waited >= cfg["cooldown_max"]:
            mon.event(f"cooldown timed out after {waited:.0f} s, going on anyway: {'; '.join(busy)}", alert=True)
            break
    mon.event(f"cooled down in {waited:.0f} s: CPU {temp} C, free RAM {s.get('mem_avail_mb')} MB")
    mon.phase = "idle"
    return {"seconds": round(waited), "temp_c": temp, "mem_avail_mb": s.get("mem_avail_mb")}


@contextlib.contextmanager
def progress_bar(name, total, unit):
    """tqdm bar on the console (not in benchmark.log); the console log lines are printed above it meanwhile."""
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    with logging_redirect_tqdm(), tqdm(total=total, desc=name, unit=unit, dynamic_ncols=True) as bar:
        yield bar


def run_worker(mon, cfg, name, stage):
    """Runs one stage of one detector in a fresh process. Returns (exit code, guard reason)."""
    mon.abort_reason, mon.uv_run = None, 0
    mon.phase = f"{name}:{stage}"
    log.info(f"{name}: {stage} worker starts")
    # own process group, so the guard and Ctrl+C stop it with everything it started
    proc = subprocess.Popen([sys.executable, "-u", str(WORKER), stage, name, cfg["config_path"]],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            bufsize=1, start_new_session=True)
    mon.proc = proc
    try:
        with contextlib.ExitStack() as bar_open:
            bar = None
            for line in proc.stdout:  # the worker's output, tracebacks included, goes to the log line by line
                if line.startswith(PROGRESS):
                    done, total = map(int, line[len(PROGRESS):].split())
                    bar = bar or bar_open.enter_context(
                        progress_bar(name, total, "frame" if stage == "fps" else "img"))
                    bar.update(done - bar.n)
                elif line.startswith(PHASE):
                    mon.phase = f"{name}:{line[len(PHASE):].strip()}"
                    log.info(f"{name}: {mon.phase.split(':', 1)[1]}")
                elif line.strip():
                    log.info(f"{name} | {line.rstrip()}")
            proc.wait()
    finally:
        if proc.poll() is None:
            stop_process(proc)
        mon.proc, mon.phase = None, "idle"
    return proc.returncode, mon.abort_reason


def describe_exit(rc, why, name):
    if why:
        return f"stopped by guard: {why}"
    if rc < 0:
        sig = signal.Signals(-rc).name
        hint = " (likely the kernel OOM killer: dmesg | grep -i oom)" if sig == "SIGKILL" else ""
        return f"killed by {sig}{hint}"
    return f"crashed with exit code {rc}, see the '{name} |' lines in benchmark.log"


def is_done(cfg, name):
    return (read_json(result_path(cfg, name)) or {}).get("status") == "ok"


def run_detector(mon, cfg, name, stages):
    """Runs the detector's worker stages in order, each in a fresh process, skipping stages finished before
    (--resume). Records the status and the telemetry aggregates in NAME.json and returns it."""
    path = result_path(cfg, name)
    res = read_json(path) or {"name": name}
    if res.get("status") == "ok":
        log.info(f"===== {name}: already done, skipped")
        return res
    log.info(f"===== {name}")
    if not res.get("done"):  # a fresh attempt: its telemetry starts now
        res["telemetry_since"] = now()
        write_json(path, res)
    for stage in stages:
        if stage in res.get("done", []):
            continue
        rc, why = run_worker(mon, cfg, name, stage)
        res = read_json(path)  # the worker added its results
        if rc != 0:
            res["status"] = f"{stage} " + describe_exit(rc, why, name)
            break
        res["done"] = res.get("done", []) + [stage]
        write_json(path, res)
    else:
        res["status"] = "ok"
    res["telemetry"] = phase_stats(cfg, name, res["telemetry_since"])
    write_json(path, res)
    mon.event(f"{name}: {res['status']}", alert=res["status"] != "ok")
    return res
