# Detector accuracy: Raspberry Pi 5 Model B Rev 1.0

- Test set: `/home/rpi/project/input/VisDrone2019-DET-test-dev-500`, 500 images, 23324 boxes
- All at IoU 0.5 (pycocotools): mAP50 and AR with detections of score >= 0.001, up to 500 per image; P/R/F1 at each detector's `conf`
- Pacing: 0.0 s rest after every image; paused at 70 °C until 60 °C
- Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, CPU governor ondemand, RAM 4049 MB

## Accuracy

| model | input | mAP50 | AR@500 | conf | P | R | F1 | best F1 | at conf |
|---|---|---|---|---|---|---|---|---|---|
| nanodet | 640x384 | 0.180 | 0.318 | 0.35 | 0.609 | 0.281 | 0.385 | 0.387 | 0.31 |
| yolo11 | 640x640 | 0.250 | 0.421 | 0.25 | 0.620 | 0.368 | 0.462 | 0.463 | 0.27 |
| yolo11_fp16 | 640x640 | 0.250 | 0.422 | 0.25 | 0.625 | 0.368 | 0.463 | 0.463 | 0.25 |
| yolo11_int8 | 640x640 | 0.236 | 0.411 | 0.25 | 0.668 | 0.327 | 0.439 | 0.449 | 0.19 |
| yolo26 | 640x640 | 0.244 | 0.409 | 0.25 | 0.608 | 0.368 | 0.458 | 0.459 | 0.25 |
| yolo8 | 640x640 | 0.236 | 0.402 | 0.25 | 0.632 | 0.357 | 0.456 | 0.457 | 0.25 |
| yolo_fastestv2 | 640x384 | 0.022 | 0.089 | 0.30 | 0.127 | 0.125 | 0.126 | 0.128 | 0.34 |

## Conditions

Per detector, over its accuracy pass and COCO evaluation. RAM used: the whole system; detector MB: its worker process.

| model | status | run min | hot pauses | paused s | start °C | end °C | max °C | min MHz | min 5V | max 5V | min core V | max core V | avg RAM used | max RAM used | avg detector MB | max detector MB | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nanodet | ok | 5.5 | 13 | 235 | 64.5 | 65.5 | 70.5 | 1500 | 5.06 | 5.17 | 0.720 | 0.870 | 2268 MB (56%) | 2553 MB (63%) | 116 | 367 | - |
| yolo11 | ok | 6.5 | 22 | 357 | 67.8 | 68.3 | 69.4 | 1500 | 5.03 | 5.17 | 0.720 | 0.870 | 2407 MB (59%) | 2628 MB (65%) | 140 | 324 | - |
| yolo11_fp16 | ok | 6.4 | 22 | 348 | 67.8 | 65.5 | 70.5 | 1500 | 4.99 | 5.17 | 0.720 | 0.870 | 2359 MB (58%) | 2562 MB (63%) | 140 | 339 | - |
| yolo11_int8 | ok | 6.4 | 19 | 347 | 68.8 | 67.2 | 69.4 | 1500 | 5.06 | 5.17 | 0.720 | 0.870 | 2395 MB (59%) | 2553 MB (63%) | 157 | 325 | - |
| yolo26 | ok | 5.4 | 21 | 290 | 66.1 | 65.0 | 70.0 | 1500 | 5.02 | 5.17 | 0.720 | 0.870 | 2403 MB (59%) | 2769 MB (68%) | 140 | 322 | - |
| yolo8 | ok | 6.2 | 21 | 338 | 68.3 | 66.1 | 70.5 | 1500 | 5.07 | 5.17 | 0.720 | 0.870 | 2670 MB (66%) | 3028 MB (75%) | 158 | 283 | - |
| yolo_fastestv2 | ok | 1.4 | 3 | 61 | 65.5 | 68.8 | 69.4 | 1500 | 5.08 | 5.16 | 0.720 | 0.870 | 2901 MB (72%) | 3153 MB (78%) | 119 | 357 | - |

## Whole run

Over the whole session (all detectors and cooldowns): 64.5 → 68.8 °C (max 70.5 °C), 5 V input 4.99–5.17 V, core 0.720–0.870 V, RAM used avg 2437 MB (60%), max 3153 MB (78%), flags: none

# Detector FPS: Raspberry Pi 5 Model B Rev 1.0

- Run: `/home/rpi/project/out/performance/20261001-210803`
- Video: `/home/rpi/project/input/kyiv_drone.mp4`, 1280x676 at 25.0 FPS; 10 warm-up frames, then 500 timed frames in chunks of 50, spread evenly over the video (49 frames skipped between chunks)
- Tracker: none; detection confidence: each detector's `conf`
- Cooldown to 60.0 °C (3 readings in a row) before every detector, and again before each chunk
- Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, CPU governor ondemand, RAM 4049 MB

## Speed

Warm-up frames: untimed, at the start of the video. Detector FPS and mean ms: `det.detect()` (preprocess + inference + postprocess) over the timed frames, timed like the pipeline's `det.timing`. Pipeline FPS: frames per second of `pipeline.iter_frames()` (offline) over the timed frames, including video decoding and the tracker. (!): throttling or under-voltage during the video, so that FPS is not clean.

| model | input | warm-up frames | frames | detector FPS | mean ms | pipeline FPS | peak RAM MB |
|---|---|---|---|---|---|---|---|
| nanodet | 640x384 | 10 | 500 | 21.48 | 46.6 | 20.63 | 127 |
| yolo11 | 640x640 | 10 | 500 | 12.92 | 77.4 | 12.59 | 162 |
| yolo11_fp16 | 640x640 | 10 | 500 | 13.11 | 76.3 | 12.77 | 162 |
| yolo11_int8 | 640x640 | 10 | 500 | 12.57 | 79.5 | 12.26 | 172 |
| yolo26 | 640x640 | 10 | 500 | 13.91 | 71.9 | 13.49 | 161 |
| yolo8 | 640x640 | 10 | 500 | 13.14 | 76.1 | 12.79 | 172 |
| yolo_fastestv2 | 640x384 | 10 | 500 | 56.12 | 17.8 | 50.56 | 126 |

## Conditions

Per detector, from its worker's start (model load, warm-up) to the end of the video. Chunk start / end °C: CPU temperature at each chunk's first and last timed frame. RAM used: the whole system; detector MB: its worker process.

| model | status | run min | chunks | chunk start °C | chunk end max °C | start °C | end °C | max °C | min MHz | min 5V | max 5V | min core V | max core V | avg RAM used | max RAM used | avg detector MB | max detector MB | flags | cooldown after s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nanodet | ok | 4.4 | 10 | 58.4–60.0 | 69.4 | 65.5 | 66.7 | 69.4 | 1500 | 5.09 | 5.17 | 0.720 | 0.870 | 2858 MB (71%) | 3118 MB (77%) | 122 | 128 | - | 38 |
| yolo11 | ok | 6.2 | 10 | 57.9–60.0 | 73.8 | 62.2 | 71.6 | 73.2 | 1500 | 5.04 | 5.16 | 0.720 | 0.870 | 2821 MB (70%) | 2948 MB (73%) | 153 | 163 | - | 30 |
| yolo11_fp16 | ok | 5.5 | 10 | 59.0–60.0 | 73.8 | 60.0 | 72.7 | 72.7 | 1500 | 4.99 | 5.17 | 0.720 | 0.870 | 2919 MB (72%) | 2957 MB (73%) | 153 | 163 | - | 30 |
| yolo11_int8 | ok | 4.8 | 10 | 59.0–60.0 | 72.7 | 63.4 | 70.0 | 71.6 | 1500 | 5.05 | 5.16 | 0.720 | 0.870 | 2934 MB (72%) | 2977 MB (74%) | 171 | 173 | - | 30 |
| yolo26 | ok | 8.7 | 10 | 59.0–60.0 | 74.9 | 60.6 | 71.6 | 72.7 | 1500 | 5.05 | 5.16 | 0.720 | 0.870 | 2906 MB (72%) | 2939 MB (73%) | 153 | 162 | - | 41 |
| yolo8 | ok | 12.6 | 10 | 59.0–60.0 | 73.8 | 65.5 | 71.6 | 73.8 | 1500 | 5.04 | 5.17 | 0.720 | 0.870 | 2832 MB (70%) | 2956 MB (73%) | 162 | 173 | - | 30 |
| yolo_fastestv2 | ok | 1.6 | 10 | 58.4–60.0 | 66.1 | 60.0 | 59.5 | 65.0 | 1500 | 5.07 | 5.16 | 0.720 | 0.870 | 2769 MB (68%) | 2776 MB (69%) | 126 | 127 | - | - |

## Whole run

Over the whole session (all detectors and cooldowns): 68.3 → 59.5 °C (max 73.8 °C), 5 V input 4.99–5.17 V, core 0.720–0.870 V, RAM used avg 2859 MB (71%), max 3118 MB (77%), flags: none
