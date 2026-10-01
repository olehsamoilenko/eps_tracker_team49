# Detector benchmarks on Raspberry Pi

Two frameworks test the detectors in `models/*_ncnn` (yolo8, yolo11, yolo26, nanodet) through the tracker's own
pipeline, the way `stream.py` processes a video file: `pipeline.load_detector`, `camera.open_source`, then
`pipeline.iter_frames` in offline mode (every frame read, then detected), which calls `det.detect`.

| | `accuracy.py`: how good | `performance.py`: how fast |
|---|---|---|
| measures | at IoU 0.5: mAP50, AR (recall with all detections), P / R / F1 at the deployment confidence | detector FPS and mean latency; pipeline FPS with video decoding; peak RAM |
| on | VisDrone2019-DET test-dev, never used in training: 500 of its 1610 images, with the same class ratio | a video: `--frames` (100) timed frames after `--warmup` (10) |
| how | slowly, so the Pi never overheats: a rest after every image, and a pause whenever the CPU reaches 70 °C, until it is back at 60 °C | each detector in one go over the video, no pauses; the Pi cools down to its idle baseline only after a video, before the next detector |
| logs per detector | average and max CPU temperature, RAM used (system and detector process), voltages, pauses | start / end / max temperature, min / max voltages, average / max RAM, throttling during the video |
| duration (estimate, Pi 5) | about 10 minutes per detector | about 10 to 30 minutes, mostly cooldowns |

Shared by both:

- **Separate processes.** Every detector stage runs in its own process (`worker.py`), so the detector's RAM goes
  back to the OS when the process exits.
- **Telemetry.** A monitor writes the Pi's state every second: CPU temperature, ARM clock, CPU load, RAM, swap,
  detector process RAM, core and 5 V voltages, fan speed, and the throttling / under-voltage flags.
- **Guard.** It stops a detector when the CPU reaches 82 °C, free RAM drops below 150 MB, or the Pi under-volts
  twice in a row. It does this instead of letting the Pi reboot.
- **Crash-safe files.** The log, telemetry and results are fsync'ed as they are written, so they survive a crash.
  After a crash, `--resume` continues where the run stopped.

Code: `accuracy.py` and `performance.py` are the entry points. They share `runner.py` (run folder, monitor,
cooldown, workers), `worker.py`, `telemetry.py`, `report.py`, `results.py`, `metrics.py` and `dataset.py`.
`subset.py` makes the 500-image test set once.

## 1. Prepare the Pi (once)

**Hardware.** Raspberry Pi 5 (4 GB or 8 GB) with:

- the official 27 W USB-C power supply (5.1 V, 5 A); weaker supplies are the usual cause of under-voltage
  reboots under load;
- the Active Cooler or a case fan.

**Software.** Use the pipeline's venv and add pycocotools and tqdm (the only extra packages):

```bash
source ~/yolo/bin/activate
pip install pycocotools tqdm
python -c "import ncnn, cv2, yaml, numpy, pycocotools, tqdm; print('ok')"
```

## 2. Inputs (once)

Both inputs live in the repo's git-ignored `input/` folder:

- `input/VisDrone2019-DET-test-dev-500/{images,annotations}`: the test set, for `accuracy.py`. It is made from the
  full test set `input/VisDrone2019-DET-test-dev` (1610 images) by `subset.py`, see below.
- `input/fps_clip.mp4`: the default video for `performance.py`. It is the VisDrone clip `uav0000140_01590_v.mp4`
  (256 frames), scaled to 1280×720 and encoded as H.264. It is enough for the defaults (10 warm-up + 100 timed
  frames). For longer runs, put your own longer video there, or pass `--video`.

**Copy from the PC** where they were prepared. The archive unpacks in place at the repo root:

```bash
# on the PC, in EPS2026/
scp VisDrone2019-DET-test-dev.tar VisDrone2019-DET-test-dev.tar.sha256 pi@<pi-ip>:eps_tracker_team49/
scp eps_tracker_team49/input/fps_clip.mp4 pi@<pi-ip>:eps_tracker_team49/input/
# on the Pi
cd eps_tracker_team49
sha256sum -c VisDrone2019-DET-test-dev.tar.sha256   # VisDrone2019-DET-test-dev.tar: OK
tar xf VisDrone2019-DET-test-dev.tar && rm VisDrone2019-DET-test-dev.tar*
ls input/VisDrone2019-DET-test-dev/images | wc -l   # 1610
```

**Or download the test set on the Pi:**

```bash
cd eps_tracker_team49 && mkdir -p input && cd input
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip
unzip -q VisDrone2019-DET-test-dev.zip && rm VisDrone2019-DET-test-dev.zip && cd ..
```

**Then make the 500-image test set** (one second, no extra disk space: the files are hard links):

```bash
python test_framework/subset.py           # -> input/VisDrone2019-DET-test-dev-500
ls input/VisDrone2019-DET-test-dev-500/images | wc -l   # 500
```

It keeps the full set's mix, so the mAP stays representative of the whole test set:

- **Class ratio.** Every class has 500/1610 of its boxes in the full set (counted after the filtering in section 9),
  so the share of each class and the boxes per image (46.6) are those of the full set.
- **Scenes.** Every image group (a `9999xxx` set, or the single frames of the `0000xxx` video clips) gives 500/1610
  of its images.
- **Same images every time.** The pick has no randomness, so the PC and the Pi get the same 500 images. Other
  sizes: `-n N`, written to `input/VisDrone2019-DET-test-dev-N`.

To run on the full test set instead: `accuracy.py --data input/VisDrone2019-DET-test-dev` (about 30 minutes per
detector).

Any video works for `performance.py` with `--video`, as long as it has at least `--warmup` + `--frames` frames.
Before the run starts, the script reads that many frames through the pipeline's own `camera.open_source` to check
this. The video is never looped. Its resolution matters: bigger frames cost more decoding and resizing.

## 3. Before every session

```bash
# steady clock: no frequency scaling during timed runs (resets to ondemand at reboot)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# nothing else running: no stream.py, no desktop (Raspberry Pi OS Lite, or connect over SSH), no apt in background
top                                          # load should be ~0

# power and temperature have been fine since boot: expect throttled=0x0
vcgencmd get_throttled

# run inside tmux, so a dropped SSH connection does not kill the run
tmux new -s bench                            # detach: Ctrl+B then D; back: tmux attach -t bench
```

## 4. Accuracy: `accuracy.py`

Run from the repo root, in the venv:

```bash
python test_framework/accuracy.py --limit 50 --pause 0 --out out/accuracy/smoke   # smoke test, a few minutes
python test_framework/accuracy.py                                                 # all detectors, full test set
python test_framework/accuracy.py --models yolo26 nanodet                         # some of them
python test_framework/accuracy.py --resume out/accuracy/20261001-153000           # after Ctrl+C, guard or crash
python test_framework/accuracy.py --resume out/accuracy/20261001-153000 --models yolo8   # add a detector to that run
```

Per detector:

1. **Accuracy pass.** The worker loads the detector as the pipeline does and runs `pipeline.iter_frames` (offline)
   over all test images, keeping detections with score ≥ 0.001. A tqdm bar on the console shows the images done,
   the speed and the time left (it is not written to `benchmark.log`).
2. **Pacing.** After every image it rests `--pause` seconds. When the CPU reaches `--pause-temp`, it stops until
   the CPU is back at `--resume-temp`. While paused, the log and telemetry show the stage `paused`.
3. **COCO evaluation.** A second worker computes the metrics with pycocotools.
4. **Log line.** The log gets the detector's average and max CPU temperature and RAM, for example:
   `yolo26: CPU avg 61.3 C, max 69.8 C; RAM used avg 1840 MB (23%), max 2410 MB (30%); detector process avg 310 MB, max 1190 MB`.

| option | default | |
|---|---|---|
| `--data DIR` | `input/VisDrone2019-DET-test-dev-500` | VisDrone folder with `images/` and `annotations/` |
| `--limit N` | all | use N images spread over all sequences |
| `--pause S` | 0.5 | rest after every image, seconds |
| `--pause-temp C`, `--resume-temp C` | 70, 60 | pause at this CPU temperature, continue at that one |
| `--eval-conf X` | 0.001 | confidence for mAP |
| `--max-det N` | 500 | max boxes per image for every detector (overrides the detectors' `MAX_DET`) |
| `--conf X` | pipeline default | confidence for P / R / F1 (yolo* 0.25, nanodet* 0.35) |

## 5. FPS: `performance.py`

```bash
python test_framework/performance.py                               # all detectors, 100 frames of input/fps_clip.mp4
python test_framework/performance.py --models yolo26 nanodet
python test_framework/performance.py --tracker bytetrack           # with the tracker behind the detector
python test_framework/performance.py --video input/long_flight.mp4 --frames 1000
```

1. **Baseline.** The Pi's idle temperature, RAM and CPU load are measured first (`--baseline-s`).
2. **Per detector, in one pass over the video:**
   - load the detector;
   - open the video with `camera.open_source`, as `stream.py -s VIDEO` does;
   - run `pipeline.iter_frames` in offline mode: every frame is read, then detected;
   - leave the first `--warmup` frames untimed, then time `det.detect()` on the next `--frames` frames.

   Nothing pauses during the video, so the Pi heats up as it would in use.
3. **Cooldown.** After each video the Pi idles until the temperature is within `--cool-delta` of the baseline. Free
   RAM and CPU load must also be back, and no throttling flag may be set. Only then does the next detector start.
   With `--start-temp C` the target is C °C instead, and the Pi also cools down to it before the first detector.
   Without a fan the Pi idles warm: give it a target it reaches at idle (a plateau above it also ends the wait).

| option | default | |
|---|---|---|
| `--video FILE` | `input/fps_clip.mp4` | video to measure on |
| `--frames N` | 100 | timed frames, right after the warm-up ones |
| `--warmup N` | 10 | untimed frames at the start of the video |
| `--tracker NAME` | none | tracker behind the detector (`tracker.TRACKERS`) |
| `--conf X` | pipeline default | detection confidence (yolo* 0.25, nanodet* 0.35) |
| `--baseline-s S` | 30 | idle seconds measured at start |
| `--cool-delta C` | 3 | cooled down at ≤ baseline temperature + C |
| `--start-temp C` | off | cool down to C °C before every detector, the first one included (instead of `--cool-delta`) |
| `--cooldown-min S`, `--cooldown-max S` | 30, 600 | cooldown bounds; after the max it goes on with a warning |
| `--ram-tolerance-mb MB` | 150 | free RAM must be back within this of the baseline |
| `--drop-caches` | off | drop the Linux page cache after each detector (needs passwordless sudo) |

Common to both frameworks:

| option | default | |
|---|---|---|
| `--models NAME ...` | all `models/*_ncnn` | detectors, in this order; a new `models/NAME_ncnn` is picked up automatically |
| `--out DIR` | `out/accuracy/<date-time>` or `out/performance/<date-time>` | results folder |
| `--resume DIR` | | continue a run with its saved settings; finished detectors are skipped. With `--models`, those detectors are added to the run, so their results go into the same `summary.md` |
| `--interval S` | 1 | telemetry period |
| `--max-temp C`, `--min-free-mb MB` | 82, 150 | guard limits |
| `--no-undervolt-guard` | | keep running under under-voltage (the run is then likely to reboot the Pi) |

## 6. Results

Everything goes to the run folder (`out/accuracy/...` or `out/performance/...`, git-ignored):

| file | content |
|---|---|
| `summary.md` | the result tables, the per-detector conditions, and a line for the whole session. `accuracy.py` rewrites it after every detector, so it holds the finished ones during the run |
| `summary.csv` | the same, one row per detector |
| `benchmark.log` | the full timestamped log: settings, every stage of every detector, the pipeline's stats lines, temperature pauses, telemetry status every 15 s, warnings, guard actions, errors |
| `telemetry.csv` | one row per second: `time, elapsed_s, phase, temp_c, arm_mhz, cpu_pct, load1, mem_used_mb, mem_avail_mb, swap_used_mb, worker_rss_mb, core_v, ext5v_v, fan_rpm, throttled, flags`. `phase` is e.g. `yolo26:accuracy`, `yolo26:paused`, `nanodet:video` or `cooldown after nanodet`; `ext5v_v` is the Pi 5's 5 V input |
| `NAME.json` | all numbers of one detector, including telemetry aggregates per stage and package versions |
| `NAME_dets.npy` | (accuracy) raw detections (`image_id, x1, y1, x2, y2, score, class`), to recompute metrics on a PC |
| `run_config.json`, `environment.json`, `gt.json` | settings, Pi model / OS / governor, and the ground truth in COCO format |

To copy the results to a PC: `rsync -av pi@<pi-ip>:eps_tracker_team49/out/ ./pi-results/`

### Example output

These show the format of `summary.md`; they are **not results**:

- The accuracy and FPS numbers come from smoke runs on a MacBook (M3 Pro): 40 test images, and 100 frames of
  `input/fps_clip.mp4`. A Pi is much slower, and the full test set gives different mAP.
- The Pi columns (temperatures, voltages, RAM) come from simulated readings, because a Mac has none.

#### `accuracy.py`

> # Detector accuracy: Raspberry Pi 5 Model B Rev 1.0
>
> - Test set: `input/VisDrone2019-DET-test-dev`, 40 images, 1796 boxes (--limit subset)
> - All at IoU 0.5 (pycocotools): mAP50 and AR with detections of score >= 0.001, up to 500 per image; P/R/F1 at each detector's `conf`
> - Pacing: 0.05 s rest after every image; paused at 70 °C until 60 °C
> - &lt;OS&gt;, kernel &lt;version&gt;, CPU governor performance, RAM 8052 MB

**Accuracy**

| model | input | mAP50 | AR@500 | conf | P | R | F1 | best F1 | at conf |
|---|---|---|---|---|---|---|---|---|---|
| nanodet | 640x384 | 0.214 | 0.355 | 0.35 | 0.582 | 0.382 | 0.461 | 0.466 | 0.33 |
| yolo11 | 640x640 | 0.254 | 0.421 | 0.25 | 0.577 | 0.470 | 0.518 | 0.524 | 0.30 |
| yolo26 | 640x640 | 0.290 | 0.426 | 0.25 | 0.586 | 0.473 | 0.523 | 0.529 | 0.27 |
| yolo8 | 640x640 | 0.289 | 0.438 | 0.25 | 0.610 | 0.468 | 0.530 | 0.532 | 0.28 |

**Conditions** (per detector, over its accuracy pass and COCO evaluation)

| model | status | run min | hot pauses | paused s | start °C | end °C | max °C | min MHz | min 5V | max 5V | min core V | max core V | avg RAM used | max RAM used | avg detector MB | max detector MB | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nanodet | ok | 0.1 | 0 | 0 | 66.0 | 70.0 | 70.0 | 2400 | 5.05 | 5.05 | 0.880 | 0.880 | 1414 MB (18%) | 1428 MB (18%) | 298 | 330 | - |
| yolo11 | ok | 0.1 | 0 | 0 | 71.0 | 73.0 | 73.0 | 2400 | 5.05 | 5.05 | 0.880 | 0.880 | 1414 MB (18%) | 1421 MB (18%) | 290 | 330 | - |
| yolo26 | ok | 0.1 | 0 | 0 | 65.0 | 68.0 | 68.0 | 2400 | 5.05 | 5.05 | 0.880 | 0.880 | 1412 MB (18%) | 1428 MB (18%) | 280 | 330 | - |
| yolo8 | ok | 0.1 | 0 | 0 | 69.0 | 71.0 | 71.0 | 2400 | 5.05 | 5.05 | 0.880 | 0.880 | 1416 MB (18%) | 1428 MB (18%) | 290 | 330 | - |

**Whole run:** 50.0 → 71.0 °C (max 73.0 °C), 5 V input 5.05–5.15 V, core 0.720–0.880 V, RAM used avg
1382 MB (17%), max 1428 MB (18%), flags: none

#### `performance.py`

> # Detector FPS: Raspberry Pi 5 Model B Rev 1.0
>
> - Video: `input/fps_clip.mp4`, 1280x720 at 30.0 FPS; 10 warm-up frames, then 100 timed frames in one go
> - Tracker: none; detection confidence: each detector's `conf`
> - Cooldown to the idle baseline (+3.0 °C) after each video, never during one
> - &lt;OS&gt;, kernel &lt;version&gt;, CPU governor performance, RAM 8052 MB

**Speed**

| model | input | warm-up frames | frames | detector FPS | mean ms | pipeline FPS | peak RAM MB |
|---|---|---|---|---|---|---|---|
| nanodet | 640x384 | 10 | 100 | 97.34 | 10.3 | 88.68 | 271 |
| yolo11 | 640x640 | 10 | 100 | 57.57 | 17.4 | 54.00 | 347 |
| yolo26 | 640x640 | 10 | 100 | 64.30 | 15.6 | 60.01 | 342 |
| yolo8 | 640x640 | 10 | 100 | 71.52 (!) | 14.0 | 66.12 | 326 |

**Conditions** (per detector, from its worker's start to the end of the video)

| model | status | run min | start °C | end °C | max °C | min MHz | min 5V | max 5V | min core V | max core V | avg RAM used | max RAM used | avg detector MB | max detector MB | flags | cooldown after s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nanodet | ok | 0.0 | 68.0 | 68.0 | 68.0 | 2400 | 5.05 | 5.05 | 0.880 | 0.880 | 1421 MB (18%) | 1421 MB (18%) | 250 | 250 | - | 2 |
| yolo11 | ok | 0.0 | 71.0 | 72.0 | 72.0 | 2400 | 5.05 | 5.05 | 0.880 | 0.880 | 1410 MB (18%) | 1414 MB (18%) | 270 | 290 | - | 2 |
| yolo26 | ok | 0.0 | 66.0 | 67.0 | 67.0 | 2400 | 5.05 | 5.05 | 0.880 | 0.880 | 1404 MB (17%) | 1407 MB (18%) | 310 | 330 | - | 2 |
| yolo8 | ok | 0.0 | 70.0 | 71.0 | 71.0 | 2400 | 5.05 | 5.05 | 0.880 | 0.880 | 1414 MB (18%) | 1428 MB (18%) | 290 | 330 | THROTTLED | - |

**Whole run:** 50.0 → 71.0 °C (max 72.0 °C), 5 V input 5.05–5.15 V, core 0.720–0.880 V, RAM used avg
1132 MB (14%), max 1428 MB (18%), flags: THROTTLED

The simulated `THROTTLED` flag during yolo8's video is what puts **(!)** next to its FPS: that number is not clean.

## 7. What the numbers mean

**Accuracy** (`accuracy.py`)

| number | how it is measured |
|---|---|
| mAP50, AR | pycocotools over the test set at IoU 0.5, with detections of score ≥ `--eval-conf`, up to `--max-det` per image for every detector. AR is the recall when all those detections are kept |
| P, R, F1 | at IoU 0.5, keeping detections with score ≥ the detector's `conf`. "best F1 / at conf" is the confidence that would maximise F1 |
| hot pauses, paused s | how often and how long the pass waited for the CPU to cool down |

**Speed** (`performance.py`)

| number | how it is measured |
|---|---|
| warm-up frames | the untimed frames at the start of the video (`--warmup`, 10 by default), kept out of the FPS so that one-off start-up costs (first allocations, cold caches) do not skew it |
| frames | timed frames (`--frames`) |
| detector FPS, mean ms | every `det.detect()` call on the timed frames, timed like the pipeline's own `det.timing`. Decoding the video is not included |
| pipeline FPS | frames per second of `pipeline.iter_frames()` (offline) over the timed frames, including video decoding and the tracker |
| peak RAM MB | peak RSS of the detector process |

**Conditions** (both, per detector, over its run; cooldowns excluded)

| number | meaning |
|---|---|
| run min | duration of the detector's run (accuracy: pass + evaluation; performance: load + warm-up + video) |
| start / end / max °C | CPU temperature at the start and at the end of the run, and its peak |
| min MHz | the lowest ARM clock; below the normal 2400 MHz means the firmware throttled |
| min / max 5V | the Pi 5's 5 V input (empty on a Pi 4); a sag under load means a weak supply or cable |
| min / max core V | the CPU core supply |
| avg / max RAM used | RAM use of the whole system (MB and % of total) |
| avg / max detector MB | RAM of the detector's worker process |
| flags | throttling flags seen during the run. In `performance.py`, **(!)** next to the FPS means throttling or under-voltage happened during the video, so that FPS is not clean: improve cooling or power and rerun |
| cooldown after s | (performance) how long the Pi idled after this detector's video |
| Whole run | the same over the entire session, from the first telemetry row to the last |

## 8. If the Pi crashes or the guard stops a detector

The **status** column and `benchmark.log` show what happened:

| status | meaning |
|---|---|
| `... stopped by guard: CPU 82.3 C >= --max-temp 82` | overheating: add or check the cooler. For `accuracy.py`, also lower `--pause-temp` or raise `--pause` |
| `... stopped by guard: under-voltage ...` | the 5 V supply sags under load: use the official 27 W supply and a short, thick cable, and unplug USB devices |
| `... stopped by guard: free RAM ...` or `killed by SIGKILL` | out of memory: close other programs. The COCO evaluation is the biggest consumer. On a 2 GB Pi use `--eval-conf 0.01`, and compare only runs with the same setting |
| `... crashed with exit code N` | a Python error: the `NAME \|` lines in `benchmark.log` have the traceback |

Throttling flags (`vcgencmd get_throttled`):

| flag | meaning |
|---|---|
| `UNDERVOLT` | the 5 V input is too low right now |
| `FREQ_CAP` | the ARM clock is capped |
| `THROTTLED` | the ARM clock is being reduced right now |
| `SOFT_TEMP_LIMIT` | the firmware's soft temperature limit is active |

If the Pi rebooted anyway:

```bash
tail -n 20 out/<framework>/<run>/telemetry.csv     # the last readings before the crash: temp, 5V, free RAM, flags
tail -n 30 out/<framework>/<run>/benchmark.log     # what was running
journalctl -b -1 -k | grep -iE "undervoltage|thermal|oom"   # kernel messages of the previous boot (if the journal is persistent)
python test_framework/accuracy.py --resume out/accuracy/<run>   # or performance.py --resume out/performance/<run>
```

## 9. Fairness notes

- **What is the same for every detector:**
  - the same Python pipeline code that `stream.py` runs;
  - the same ncnn thread count (4);
  - in `accuracy.py`: the same images, ground truth, evaluation code and max boxes per image;
  - in `performance.py`: the same video and warm-up, and the same starting state of the Pi;
  - telemetry costs about 4 `vcgencmd` calls per second, during every run alike.
- **What is not the same:**
  - Input sizes follow each model's `metadata.yaml`: YOLO 640×640, NanoDet 640×384.
  - NMS IoU is each detector's own (`IOU`: YOLO 0.7, NanoDet 0.6).
  - Confidence is each family's deployment default. mAP does not depend on it.
- **Ground-truth filtering:** VisDrone ignored regions and the `others` class are removed, exactly as in training. The
  mAP is therefore comparable with the training validation numbers, not with the official VisDrone toolkit.
- **Pipeline sanity check:** running on the validation split should land close to the training validation mAP50.
  Download `VisDrone2019-DET-val.zip` from the same URL and run
  `python test_framework/accuracy.py --data input/VisDrone2019-DET-val --out out/accuracy/val-check`.
  Use it only as a check, never as the reported result: `val` was used to pick the checkpoints.
