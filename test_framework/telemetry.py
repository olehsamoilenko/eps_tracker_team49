"""Raspberry Pi telemetry and the overload guard.

Monitor samples CPU temperature, ARM clock, CPU load, RAM, swap, the worker's RSS, core and 5 V rail voltages, fan
speed and the firmware throttling flags into telemetry.csv (fsync'ed every row, so the last readings before a crash
survive it), logs flag changes and a status line every STATUS_EVERY s, and stops the running worker before the Pi
overheats, runs out of RAM or browns out. Missing sources (not a Pi; a Pi 4 has no PMIC 5 V reading) give empty cells.
"""
import csv
import glob
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import threading
import time

from results import now

log = logging.getLogger("benchmark")
VCGENCMD = shutil.which("vcgencmd")
# vcgencmd get_throttled: bits 0-3 = happening now, bits 16-19 = happened since boot
FLAGS = {1: "UNDERVOLT", 2: "FREQ_CAP", 4: "THROTTLED", 8: "SOFT_TEMP_LIMIT"}
IDLE_CPU_PCT = 15  # CPU load above this is "not idle"
STATUS_EVERY = 15  # seconds between status lines in the log


def read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except (OSError, TypeError):
        return None


def vcgencmd(*args):
    if not VCGENCMD:
        return ""
    try:
        return subprocess.run([VCGENCMD, *args], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def num(pattern, text, scale=1.0):
    m = re.search(pattern, text or "")
    return round(float(m.group(1)) * scale, 3) if m else None


def flag_names(bits):
    return "|".join(n for b, n in FLAGS.items() if bits & b)


def throttled():
    m = re.search(r"0x[0-9a-fA-F]+", vcgencmd("get_throttled"))
    return int(m.group(0), 16) if m else None


def cpu_temp():
    t = read_text("/sys/class/thermal/thermal_zone0/temp")
    return round(int(t) / 1000, 1) if t else None


class Sampler:
    """One telemetry reading from /proc, /sys and vcgencmd."""

    def __init__(self):
        self.fan = next(iter(glob.glob("/sys/devices/platform/cooling_fan/hwmon/*/fan1_input")), None)
        self.cpu_prev = self._cpu_times()

    @staticmethod
    def _cpu_times():
        f = (read_text("/proc/stat") or "").split()
        if f[:1] != ["cpu"]:
            return None
        v = [int(x) for x in f[1:9]]
        return sum(v), v[3] + v[4]  # total, idle + iowait

    def _cpu_pct(self):
        cur, prev = self._cpu_times(), self.cpu_prev
        self.cpu_prev = cur
        if not cur or not prev or cur[0] == prev[0]:
            return None
        return round(100 * (1 - (cur[1] - prev[1]) / (cur[0] - prev[0])), 1)

    def sample(self, pid=None):
        mem = {}
        for line in (read_text("/proc/meminfo") or "").splitlines():
            key, _, val = line.partition(":")
            mem[key] = int(val.split()[0]) / 1024  # kB -> MB
        temp = cpu_temp()
        # the real ARM clock (drops when the firmware throttles); sysfs only shows the requested one
        mhz = num(r"=(\d+)", vcgencmd("measure_clock", "arm"), 1e-6)
        if mhz is None:
            mhz = num(r"(\d+)", read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"), 1e-3)
        rss = num(r"VmRSS:\s+(\d+)", read_text(f"/proc/{pid}/status"), 1 / 1024) if pid else None
        has_mem = "MemAvailable" in mem
        return {
            "temp_c": temp,
            "arm_mhz": round(mhz) if mhz else None,
            "cpu_pct": self._cpu_pct(),
            "load1": round(os.getloadavg()[0], 2),
            "mem_used_mb": round(mem["MemTotal"] - mem["MemAvailable"]) if has_mem else None,
            "mem_avail_mb": round(mem["MemAvailable"]) if has_mem else None,
            "swap_used_mb": round(mem["SwapTotal"] - mem["SwapFree"]) if "SwapFree" in mem else None,
            "worker_rss_mb": round(rss) if rss else None,
            "core_v": num(r"volt=([\d.]+)", vcgencmd("measure_volts", "core")),
            "ext5v_v": num(r"EXT5V_V volt\(\d+\)=([\d.]+)", vcgencmd("pmic_read_adc", "EXT5V_V")),  # Pi 5
            "fan_rpm": num(r"(\d+)", read_text(self.fan)) if self.fan else None,
            "throttled": throttled(),
        }


def stop_process(proc):
    """SIGTERM the worker's process group, SIGKILL if it is still alive after 5 s."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=5)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass


class Monitor(threading.Thread):
    """Writes a telemetry row every cfg['interval'] s and stops the running worker if the Pi is in danger.
    The orchestrator sets .phase (row label) and .proc (the running worker or None)."""

    FIELDS = ["time", "elapsed_s", "phase", "temp_c", "arm_mhz", "cpu_pct", "load1", "mem_used_mb",
              "mem_avail_mb", "swap_used_mb", "worker_rss_mb", "core_v", "ext5v_v", "fan_rpm",
              "throttled", "flags"]

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg, self.sampler = cfg, Sampler()
        self.phase, self.proc, self.abort_reason, self.last = "idle", None, None, {}
        self.t0, self.last_print, self.prev_bits, self.uv_run = time.time(), 0.0, 0, 0
        self.stopped = threading.Event()
        path = os.path.join(cfg["out"], "telemetry.csv")
        new = not os.path.exists(path)
        self.csv_file = open(path, "a", newline="")
        self.csv = csv.DictWriter(self.csv_file, self.FIELDS)
        if new:
            self.csv.writeheader()

    def event(self, msg, alert=False):
        (log.warning if alert else log.info)(f"[{self.phase}] {msg}")

    def run(self):
        while not self.stopped.is_set():
            try:
                self._tick()
            except Exception:  # telemetry must never break the benchmark
                log.exception("telemetry error")
            self.stopped.wait(self.cfg["interval"])

    def stop(self):
        self.stopped.set()
        self.join(timeout=15)
        self.csv_file.close()

    def _tick(self):
        proc = self.proc
        s = self.sampler.sample(proc.pid if proc else None)
        bits = (s["throttled"] or 0) & 0xF
        s.update(time=now(), elapsed_s=round(time.time() - self.t0), phase=self.phase, flags=flag_names(bits))
        self.csv.writerow(dict(s, throttled=hex(s["throttled"]) if s["throttled"] is not None else None))
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())  # the last rows before a crash are the most valuable ones
        self.last = s

        if bits & ~self.prev_bits:
            self.event(f"{flag_names(bits & ~self.prev_bits)} now (CPU {s['temp_c']} C, "
                       f"5V rail {s['ext5v_v']} V, core {s['core_v']} V)", alert=True)
        elif self.prev_bits and not bits:
            self.event("throttling flags cleared")
        self.prev_bits = bits
        if proc and proc.poll() is None:
            self._guard(s, bits, proc)
        if time.time() - self.last_print >= STATUS_EVERY:
            self.last_print = time.time()
            log.info(f"[{self.phase}] {s['temp_c']} C  {s['arm_mhz']} MHz  CPU {s['cpu_pct']}%  "
                     f"RAM free {s['mem_avail_mb']} MB  worker {s['worker_rss_mb']} MB  "
                     f"5V {s['ext5v_v']} V  {s['flags'] or 'no flags'}")

    def _guard(self, s, bits, proc):
        self.uv_run = self.uv_run + 1 if bits & 1 else 0
        c, reason = self.cfg, None
        if s["temp_c"] is not None and s["temp_c"] >= c["max_temp"]:
            reason = f"CPU {s['temp_c']} C >= --max-temp {c['max_temp']}"
        elif s["mem_avail_mb"] is not None and s["mem_avail_mb"] < c["min_free_mb"]:
            reason = f"free RAM {s['mem_avail_mb']} MB < --min-free-mb {c['min_free_mb']}"
        elif c["undervolt_guard"] and self.uv_run >= 2:
            reason = f"under-voltage for {self.uv_run} samples in a row (5V rail {s['ext5v_v']} V)"
        if reason and not self.abort_reason:
            self.abort_reason = reason
            self.event(f"GUARD stops the model: {reason}", alert=True)
            stop_process(proc)


AGGREGATES = {"temp_c": ("max", "mean"), "arm_mhz": ("min", "mean"), "cpu_pct": ("mean",),
              "mem_used_mb": ("max", "mean"), "mem_avail_mb": ("min",), "swap_used_mb": ("max",),
              "worker_rss_mb": ("max", "mean"), "ext5v_v": ("min", "max"), "core_v": ("min", "max")}


def telemetry_rows(cfg):
    with open(os.path.join(cfg["out"], "telemetry.csv"), newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows, interval):
    """Telemetry rows -> duration, start/end temperature, min/max/mean per AGGREGATES, flags seen."""
    fns = {"max": max, "min": min, "mean": lambda v: sum(v) / len(v)}
    temps = [float(r["temp_c"]) for r in rows if r["temp_c"]]
    st = {"seconds": round(len(rows) * interval),
          "start_temp_c": temps[0] if temps else None, "end_temp_c": temps[-1] if temps else None}
    for key, how in AGGREGATES.items():
        v = [float(r[key]) for r in rows if r[key]]
        for fn in how if v else ():
            st[f"{key}_{fn}"] = round(fns[fn](v), 2)
    st["flags"] = sorted({f for r in rows for f in r["flags"].split("|") if f})
    return st


def phase_stats(cfg, name, since):
    """Aggregates of one detector's telemetry rows from `since` (ISO time) on: per stage, and "run" over all of
    them (from its worker's start to the end of its last stage; cooldowns are not part of it)."""
    rows = [r for r in telemetry_rows(cfg) if r["phase"].startswith(name + ":") and r["time"] >= since]
    stages = dict.fromkeys(r["phase"].split(":", 1)[1] for r in rows)
    stats = {s: aggregate([r for r in rows if r["phase"] == f"{name}:{s}"], cfg["interval"]) for s in stages}
    if rows:
        stats["run"] = aggregate(rows, cfg["interval"])
    return stats


def environment():
    os_release = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', read_text("/etc/os-release") or "", re.M))
    gov = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    ram = num(r"MemTotal:\s+(\d+)", read_text("/proc/meminfo"), 1 / 1024)
    thr = throttled()
    return {
        "time": now(),
        "device": (read_text("/proc/device-tree/model") or platform.platform()).strip("\x00\n "),
        "os": os_release.get("PRETTY_NAME"), "kernel": platform.release(), "python": platform.python_version(),
        "cpu_governor": gov.strip() if gov else None, "ram_mb": round(ram) if ram else None,
        "throttled_since_boot": flag_names(thr >> 16) if thr is not None else None,
    }
