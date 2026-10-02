"""Video pipeline for Raspberry Pi: source -> optional detector (+ tracker) -> web stream and/or file.

    python stream.py                                               # Pi camera -> web :5000, no detection
    python stream.py -d nanodet                                    # Pi camera -> NanoDet -> web
    python stream.py -d yolo_fastestv2                             # Pi camera -> Yolo-FastestV2 -> web
    python stream.py -d yolo26 --web 8080 --save rec.mp4           # Pi camera -> YOLO26 -> web + recording
    python stream.py -d nanodet --classes pedestrian,car           # only these classes
    python stream.py -d yolo8 --tracker bytetrack                  # boxes labeled with track IDs
    python stream.py -s input/street.avi -d yolo11 --save out.mp4  # video -> video, every frame detected
    python stream.py -s input/street.avi -d nanodet                # video played in real time, looped -> web
    python stream.py -s input/0000001.jpg -d yolo8 --save out.jpg  # image -> image
    python stream.py --once -d nanodet --save snap.jpg             # one camera frame -> image
    python stream.py -d nanodet --duration 10 --save clip.mp4      # record 10 s from the camera
    python stream.py -d yolo8 --tracker bytetrack -v               # camera, detector capped: for a power bank
    python stream.py -d yolo8 --tracker bytetrack -v 4             # same, at most 4 detections per second

Detectors (-d) are the models/NAME_ncnn directories; all of them report the same VisDrone classes.
Live mode (camera, or a video with web output) runs at source speed: detection runs in its own thread and
its latest finished boxes are drawn on every new frame, so boxes lag by up to one detection.
A video with only --save is processed frame by frame: exact boxes, output at the source FPS.
An image or --once gives a single frame; with web output it is served until Ctrl+C.
--save FILE writes a video, or an image for .jpg/.png (it then holds the latest frame).
-v [N] (camera only) caps the detector at N runs per second (default LOW_POWER_DET_FPS) and idles in between,
to lower the power draw and heat on a weak supply (power bank); the camera stream keeps its full frame rate.
Run inside the ~/yolo venv. Stop with Ctrl+C.
"""
import argparse
import threading

from camera import is_image, open_source
from common import VISDRONE
from output import Recorder, draw
from pipeline import DETECTORS, iter_frames, load_detector
from stats import Stats
from tracker import TRACKERS, load_tracker

LOW_POWER_DET_FPS = 6  # -v without a number: detector runs per second (it does ~12/s on the Pi 5 uncapped)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--source", default="picam", help="picam (Pi CSI camera), video or image file")
    p.add_argument("--once", action="store_true", help="process a single frame (camera snapshot)")
    p.add_argument("--duration", type=float, metavar="SEC", help="stop after SEC seconds from the first frame")
    p.add_argument("-d", "--detector", choices=DETECTORS, help="model in models/; omit for a clean camera stream")
    p.add_argument("--conf", type=float,
                   help="confidence threshold (default: yolo_fastest* 0.3, other yolo* 0.25, nanodet* 0.35)")
    p.add_argument("--classes", help=f"keep only these classes, comma-separated: {','.join(VISDRONE)}")
    p.add_argument("--tracker", choices=TRACKERS, default="none", help="object tracker (needs a detector)")
    p.add_argument("--web", type=int, nargs="?", const=5000, metavar="PORT",
                   help="MJPEG stream for the browser, port 5000 by default (on when --save is not given)")
    p.add_argument("--save", metavar="FILE", help="write the output: video (.mp4) or image (.jpg/.png)")
    p.add_argument("--width", type=int, default=640, help="camera frame width")
    p.add_argument("--height", type=int, default=480, help="camera frame height")
    p.add_argument("--fps", type=int, default=30, help="camera frame rate")
    p.add_argument("-v", "--low-power", type=float, nargs="?", const=LOW_POWER_DET_FPS, default=0, metavar="N",
                   help=f"camera only: at most N detections per second, {LOW_POWER_DET_FPS} by default "
                        "(less power draw for a power bank)")
    args = p.parse_args()
    if args.low_power and args.source != "picam":
        p.error("-v is for the camera (-s picam) only")
    if args.tracker != "none" and not args.detector:
        p.error("--tracker needs a detector (-d)")
    if args.web is None and args.save is None:
        args.web = 5000
    return args


def main():
    args = parse_args()
    if args.low_power:
        print(f"Low-power mode: at most {args.low_power:g} detections per second", flush=True)
    det = load_detector(args.detector, args.conf, args.classes)
    tracker = load_tracker(args.tracker)
    cap, fps, is_file = open_source(args.source, args.width, args.height, args.fps)
    web = None
    if args.web:
        from web_stream import WebStream  # flask only when streaming
        web = WebStream(args.web)
    rec = Recorder(args.save, fps) if args.save else None
    single = args.once or is_image(args.source)
    frames = iter_frames(cap, fps, is_file, det, tracker, single, offline=is_file and not web,
                         max_det_fps=args.low_power)
    stats = Stats(det, (args.detector or "") + (f" + {args.tracker}" if args.tracker != "none" else ""))
    try:
        for frame, boxes in frames:
            stats.tick(boxes)
            if det:
                draw(frame, boxes, det.names)
                stats.overlay(frame)
            if web:
                web.push(frame)
            if rec:
                rec.write(frame)
            stats.log(now=single)
            if args.duration and stats.elapsed >= args.duration:
                print(f"Stopped after {args.duration:g} s, {stats.n} frames: {stats.text()}", flush=True)
                break
        else:
            print(f"Source ended after {stats.n} frames", flush=True)
        if single and web:
            print("Serving the frame, Ctrl+C to stop", flush=True)
            threading.Event().wait()
    except KeyboardInterrupt:
        print(f"Stopped after {stats.n} frames")
    finally:
        frames.close()  # stops the live-mode workers
        cap.release()
        if rec:
            rec.close()


if __name__ == "__main__":
    main()
