"""Ground truth: a VisDrone2019-DET folder (images/ + annotations/) -> COCO json for pycocotools."""
import os

from common import VISDRONE
from results import write_json


def image_files(data):
    files = sorted(f for f in os.listdir(os.path.join(data, "images")) if f.lower().endswith((".jpg", ".png")))
    if not files:
        raise SystemExit(f"No images in {data}/images")
    return files


def read_boxes(data, fname):
    """Ground truth of one image as [(class id, [x, y, w, h])]. Same filtering as in training (Ultralytics
    VisDrone.yaml, NanoDet's visdrone2coco.py), so the numbers are comparable with the validation ones."""
    boxes = []
    with open(os.path.join(data, "annotations", os.path.splitext(fname)[0] + ".txt")) as f:
        for line in f:
            if not line.strip():
                continue
            # x, y, w, h, score, category, truncation, occlusion
            x, y, w, h, score, cat = (int(v) for v in line.strip().rstrip(",").split(",")[:6])
            # score 0: ignored regions (category 0) and all "others" (category 11)
            if score == 0 or not 1 <= cat <= len(VISDRONE) or w <= 0 or h <= 0:
                continue
            boxes.append((cat - 1, [x, y, w, h]))
    return boxes


def build_gt(data, limit, path):
    """limit: that many images spread over all sequences, 0 = all. Returns (images, boxes)."""
    files = image_files(data)
    if limit and limit < len(files):
        files = [files[i * len(files) // limit] for i in range(limit)]
    images, anns = [], []
    for img_id, fname in enumerate(files, 1):
        images.append({"id": img_id, "file_name": fname})  # bbox evaluation needs no width/height
        for cat, (x, y, w, h) in read_boxes(data, fname):
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    write_json(path, {"info": {"description": os.path.basename(data)},  # pycocotools >= 2.0.9 needs info
                      "images": images, "annotations": anns,
                      "categories": [{"id": i, "name": n} for i, n in enumerate(VISDRONE)]})
    return len(images), len(anns)
