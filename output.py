"""Frame output: boxes drawn on the frame, and --save to a video or an image file."""
import cv2
import numpy as np

from camera import is_image


def color(i):
    return tuple(int(c) for c in np.random.RandomState(i).randint(64, 255, 3))


def draw(frame, boxes, names):
    for x1, y1, x2, y2, tid, conf, c in boxes:
        name = names.get(int(c), str(int(c)))
        col = color(int(tid) if tid >= 0 else int(c))
        label = f"{name} {int(tid)}" if tid >= 0 else f"{name} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)


class Recorder:
    """A video file, or an image (.jpg/.png) that holds the latest frame."""

    def __init__(self, path, fps):
        self.path, self.fps = path, fps
        self.writer, self.saved = None, False

    def write(self, frame):
        if is_image(self.path):
            self.saved = cv2.imwrite(self.path, frame)
            return
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, frame.shape[1::-1])
            if not self.writer.isOpened():
                raise SystemExit(f"Cannot write {self.path}")
        self.writer.write(frame)
        self.saved = True

    def close(self):
        if self.writer:
            self.writer.release()
        if self.saved:
            print(f"Saved {self.path}")
