"""MJPEG web stream: every browser viewer gets the newest frame at http://<pi-ip>:PORT."""
import socket
import threading

import cv2
from flask import Flask, Response
from werkzeug.serving import make_server

from common import Latest

PAGE = '<html><body style="margin:0;background:#111"><img src="/video" style="width:100%;height:auto"></body></html>'
JPEG_QUALITY = 80


class WebStream:
    def __init__(self, port):
        self.jpeg = Latest()
        app = Flask(__name__)
        app.add_url_rule("/", "index", lambda: PAGE)
        app.add_url_rule("/video", "video", lambda: Response(
            self._mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"))
        server = make_server("0.0.0.0", port, app, threaded=True)  # binds here, so a busy port fails at start
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Web stream: http://{local_ip()}:{port}", flush=True)

    def push(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            self.jpeg.put(buf.tobytes())

    def _mjpeg(self):
        seq = 0
        while True:
            jpeg, seq = self.jpeg.wait_new(seq)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


def local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))  # sends nothing, just picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return "<pi-ip>"
