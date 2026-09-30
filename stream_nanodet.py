"""MJPEG-трансляция с Raspberry Pi в браузер: камера -> NanoDet + трекинг ByteTrack -> рамки и FPS на кадре.

Камера и детектор работают в фоне ОДИН раз; все открытые окна браузера получают один и тот же
последний готовый кадр (в исходной версии каждый зритель заново открывал камеру).

    python stream_nanodet.py                                   # CSI-камера Pi (rpicam-vid), 640x480
    python stream_nanodet.py --source 0                        # USB-камера /dev/video0
    python stream_nanodet.py --model models/nanodet-plus-m_320.ncnn.param --detect-every 2
    python stream_nanodet.py --source street.avi               # проверка на видеофайле
Открыть в браузере: http://IP_МАЛИНКИ:5000
"""
import argparse
import os
import platform
import re
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from run_nanodet import NanoDet, class_name, color
from run_pi import RpicamVid

ARM = platform.machine().lower() in ("aarch64", "arm64")


class Camera:
    """Читает камеру (или видеофайл в реальном времени) в отдельном потоке и хранит только свежий кадр."""

    def __init__(self, source, width, height):
        self.is_file = os.path.exists(source)
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        elif source == "picam":  # CSI-камера Raspberry Pi (например IMX462) через rpicam-vid
            self.cap = RpicamVid(width, height)
        else:
            backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(source), backend)
            # MJPG от самой камеры: большинство USB-камер отдают 640x480@30 только в нём
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # не копить старые кадры в драйвере
            time.sleep(1.0)  # V4L2 нужно время на инициализацию
        self.frame, self.seq = None, 0
        self.cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        t0, n = time.perf_counter(), 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:  # файл крутим по кругу
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    t0, n = time.perf_counter(), 0
                else:
                    time.sleep(0.01)
                continue
            if self.is_file:  # файл отдаём с его родной частотой, как живая камера
                n += 1
                delay = t0 + n / self.fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            with self.cv:
                self.frame, self.seq = frame, self.seq + 1
                self.cv.notify_all()

    def wait_new(self, last_seq):
        """Дождаться кадра новее last_seq; промежуточные кадры пропускаются."""
        with self.cv:
            while self.seq == last_seq:
                self.cv.wait()
            return self.frame, self.seq


class Processor:
    """Фоновый поток: свежий кадр -> детекция/трекинг -> отрисовка -> JPEG."""

    def __init__(self, cam, args):
        self.cam, self.args = cam, args
        size = args.size
        if size is None:
            m = re.search(r"_(\d{3})\.(?:onnx|ncnn\.param)$", os.path.basename(args.model))
            size = f"{m.group(1)},{m.group(1)}" if m else "640,384"
        w, h = map(int, size.split(","))
        self.det = NanoDet(args.model, w, h, args.threads)
        self.tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))
        self.wanted = set(args.classes.split(",")) if args.classes else None
        self.jpeg, self.jpeg_seq = None, 0
        self.cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        a, cam_seq, n, dt_avg, t_det, n_det = self.args, 0, 0, 0.0, 0.0, 0
        t_start = time.time()
        while True:
            frame, cam_seq = self.cam.wait_new(cam_seq)
            frame = frame.copy()
            t = time.perf_counter()
            if n % a.detect_every:
                # кадр без детектора: треки сдвигает фильтр Калмана
                STrack.multi_predict(self.tracker.tracked_stracks + self.tracker.lost_stracks)
                rows = [(*tr.xyxy, tr.track_id, int(tr.cls))
                        for tr in self.tracker.tracked_stracks if tr.is_activated]
            else:
                td = time.perf_counter()
                dets = self.det(frame, conf=a.conf)
                t_det += time.perf_counter() - td
                n_det += 1
                if self.wanted is not None and len(dets):
                    dets = dets[[class_name(int(c), self.det.nc) in self.wanted for c in dets[:, 5]]]
                out = self.tracker.update(Boxes(dets, frame.shape[:2]).numpy(), frame)
                rows = [(*tr[:4], int(tr[4]), int(tr[6])) for tr in out]
            dt = time.perf_counter() - t
            dt_avg = 0.9 * dt_avg + 0.1 * dt if n else dt
            n += 1

            for x1, y1, x2, y2, tid, c in rows:
                col = color(tid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(frame, f"{class_name(c, self.det.nc)} {tid}", (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            fps = n / max(time.time() - t_start, 1e-6)
            cv2.putText(frame, f"{fps:.1f} FPS  det {t_det / max(n_det, 1) * 1000:.0f} ms  tracks {len(rows)}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            if ok:
                with self.cv:
                    self.jpeg, self.jpeg_seq = buf.tobytes(), self.jpeg_seq + 1
                    self.cv.notify_all()
            if n % 60 == 0:
                print(f"кадр {n}: {fps:.1f} FPS обработки, детектор {t_det / max(n_det, 1) * 1000:.0f} мс, "
                      f"обработка кадра {dt_avg * 1000:.0f} мс, треков {len(rows)}", flush=True)

    def frames(self):
        """Генератор для одного зрителя: отдаёт каждый новый готовый JPEG."""
        last = 0
        while True:
            with self.cv:
                while self.jpeg_seq == last:
                    self.cv.wait()
                jpeg, last = self.jpeg, self.jpeg_seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/nanodet-plus-m_416.ncnn.param" if ARM
                   else "models/nanodet-plus-m_416.onnx")
    p.add_argument("--source", default="picam" if ARM else "0",
                   help="picam (CSI-камера Pi), индекс USB-камеры (0 = /dev/video0) или видеофайл")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--size", default=None, help="вход модели ширина,высота (по умолчанию из имени *_416)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--classes", default=None, help="только эти классы: person,car")
    p.add_argument("--detect-every", type=int, default=1, help="детектор раз в N кадров, между ними трекер")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    proc = Processor(Camera(args.source, args.width, args.height), args)
    app = Flask(__name__)

    @app.route("/")
    def index():
        return ('<html><body style="margin:0;background:#111"><img src="/video" '
                'style="width:100%;height:auto"></body></html>')

    @app.route("/video")
    def video():
        return Response(proc.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"Трансляция с детекцией запущена! Откройте в браузере: http://IP_ВАШЕЙ_МАЛИНКИ:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
