"""Performance stats: FPS counter, detector timing, and the output loop's overlay text and console log."""
import time

import cv2

LOG_EVERY = 5  # seconds between console stats lines
FONT, SCALE, LINE = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 24  # overlay text: font, scale, panel line height px
PANEL_ALPHA = 0.55  # overlay background opacity
WHITE = (255, 255, 255)  # BGR


class Rate:
    """Smoothed events per second. Averages the interval, not 1/interval: jittery intervals would inflate the FPS."""

    def __init__(self):
        self.last, self.dt = None, 0.0

    def tick(self):
        now = time.perf_counter()
        if self.last is not None:
            dt = now - self.last
            self.dt = 0.9 * self.dt + 0.1 * dt if self.dt else dt
        self.last = now

    @property
    def fps(self):
        return 1 / max(self.dt, 1e-6) if self.dt else 0.0


class Timing:
    """Smoothed duration (ms) and rate of a repeated task; wrap each run in `with timing:`."""

    def __init__(self):
        self.ms, self.rate, self.t = 0.0, Rate(), 0.0

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *exc):
        ms = (time.perf_counter() - self.t) * 1000
        self.ms = 0.9 * self.ms + 0.1 * ms if self.ms else ms
        self.rate.tick()


class Stats:
    """Output frame count and FPS plus the detector's timing (det may be None)."""

    def __init__(self, det, label=""):
        self.det, self.rate, self.n, self.boxes, self.label = det, Rate(), 0, None, label
        self.t0, self.t_log = None, time.perf_counter()

    def tick(self, boxes):
        """Once per output frame, with its boxes."""
        self.rate.tick()
        self.n += 1
        self.boxes = boxes
        if self.t0 is None:
            self.t0 = time.perf_counter()

    @property
    def elapsed(self):
        """Seconds since the first frame."""
        return time.perf_counter() - self.t0

    def text(self):
        s = f"{self.rate.fps:.1f} FPS  " if self.rate.fps else ""
        if self.det:
            t = self.det.timing
            det_fps = f" ({t.rate.fps:.1f} FPS)" if t.rate.fps else ""
            s += f"det {t.ms:.0f} ms{det_fps}  objs {len(self.boxes)}"
        return s.strip()

    def overlay(self, frame):
        """Top left, on a translucent dark panel: detector + tracker and FPS (green / yellow / red by speed)."""
        fps = self.rate.fps
        fps_col = (80, 230, 80) if fps >= 15 else (0, 215, 255) if fps >= 8 else (80, 80, 255)  # BGR
        parts = [(self.label, WHITE), (f"   {fps:.0f} FPS" if fps else "", fps_col)]
        width = sum(cv2.getTextSize(t, FONT, SCALE, 1)[0][0] for t, _ in parts)
        roi = frame[:8 + LINE, :16 + width]
        roi[:] = (roi * (1 - PANEL_ALPHA)).astype(roi.dtype)  # darkens: black panel at PANEL_ALPHA opacity
        x = 8
        for text, col in parts:
            cv2.putText(frame, text, (x, LINE - 3), FONT, SCALE, col, 1, cv2.LINE_AA)
            x += cv2.getTextSize(text, FONT, SCALE, 1)[0][0]

    def log(self, now=False):
        """Prints the stats every LOG_EVERY seconds, or right away with now=True."""
        if now or time.perf_counter() - self.t_log > LOG_EVERY:
            self.t_log = time.perf_counter()
            print(f"frame {self.n}: {self.text() or 'no detector'}", flush=True)
