"""Frame sources with the cv2.VideoCapture read()/release() interface: Pi CSI camera, video or image file."""
import os
import subprocess

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image(path):
    return path.lower().endswith(IMAGE_EXTS)


class RpicamVid:
    """Raspberry Pi CSI camera via rpicam-vid: raw YUV420 frames over a pipe.
    Works in any venv (picamera2 exists only in the system Python). Same interface as cv2.VideoCapture."""

    def __init__(self, width, height, fps):
        self.w, self.h = width, height
        self.size = width * height * 3 // 2
        self.proc = subprocess.Popen(
            ["rpicam-vid", "-t", "0", "-n", "--codec", "yuv420", "--width", str(width),
             "--height", str(height), "--framerate", str(fps), "-o", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def read(self):
        buf = bytearray()
        while len(buf) < self.size:
            chunk = self.proc.stdout.read(self.size - len(buf))
            if not chunk:
                return False, None
            buf += chunk
        yuv = np.frombuffer(buf, np.uint8).reshape(self.h * 3 // 2, self.w)
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def release(self):
        self.proc.terminate()


class ImageFile:
    """A still image as a one-frame source."""

    def __init__(self, path):
        self.frame = cv2.imread(path)

    def read(self):
        frame, self.frame = self.frame, None
        return frame is not None, frame

    def release(self):
        pass


def open_source(spec, width, height, fps):
    """spec: 'picam', a video or an image file; width/height/fps apply to the camera. Returns (capture, fps, is_file)."""
    if os.path.isfile(spec):
        if is_image(spec):
            return ImageFile(spec), 1, True
        cap = cv2.VideoCapture(spec)
        return cap, cap.get(cv2.CAP_PROP_FPS) or 30, True
    if spec == "picam":
        return RpicamVid(width, height, fps), fps, False
    raise SystemExit(f"Source not found: {spec} (expected picam, a video or an image)")
