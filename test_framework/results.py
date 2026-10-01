"""Run folder files: layout, crash-safe JSON, the run log, and the worker -> orchestrator phase marker.

A run folder (out/benchmark/<date-time>) holds:
  benchmark.log    everything printed during the run, timestamped (workers' output too)
  telemetry.csv    one row per second: temperature, clock, RAM, voltages, throttling flags, phase
  NAME.json        all results of one detector; NAME_dets.npy its raw detections from the accuracy pass
  run_config.json, environment.json, gt.json
The results tables of all detectors go to the one summary file, SUMMARY (report.py).
"""
import datetime
import json
import logging
import os
import sys

PHASE = "##PHASE "  # a worker prints PHASE + stage when a stage starts; the orchestrator labels telemetry with it
PROGRESS = "##PROGRESS "  # PROGRESS + "done total" after every image: the orchestrator's console progress bar
# accuracy.py and performance.py each keep their part of it up to date with their latest run
SUMMARY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "summary.md")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_json(path, obj):
    """Atomic and fsync'ed: a crash mid-write must not destroy finished results."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def result_path(cfg, name):
    return os.path.join(cfg["out"], f"{name}.json")


def dets_path(cfg, name):
    return os.path.join(cfg["out"], f"{name}_dets.npy")


class SyncedFileHandler(logging.FileHandler):
    """fsync after every record: the last lines before a crash or a power loss are the most valuable ones."""

    def emit(self, record):
        super().emit(record)
        os.fsync(self.stream.fileno())


def setup_logging(out):
    """Log to the console and to out/benchmark.log (appended on --resume)."""
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(message)s", "%H:%M:%S"))
    file = SyncedFileHandler(os.path.join(out, "benchmark.log"))
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[console, file])
