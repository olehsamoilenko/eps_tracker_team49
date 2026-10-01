"""Accuracy and speed numbers: COCO mAP50 (pycocotools), P/R/F1 at a confidence, latency statistics."""
import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def latency_stats(ms):
    a = np.asarray(ms, dtype=np.float64)

    def r(v):
        return round(float(v), 2)

    return {"n": int(a.size), "fps": r(1000 / a.mean()), "mean_ms": r(a.mean()), "std_ms": r(a.std()),
            "min_ms": r(a.min()), "max_ms": r(a.max())}


def quiet():
    return contextlib.redirect_stdout(io.StringIO())  # pycocotools prints a lot


def coco_metrics(gt_path, dets, max_det, conf):
    """dets: (N, 7) [image_id, x1, y1, x2, y2, score, class] -> metrics at IoU 0.5: mAP50, AR (recall with all
    detections, up to max_det per image), and P / R / F1 at conf."""
    with quiet():
        gt = COCO(gt_path)
    anns = [{"image_id": int(i), "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s}
            for i, x1, y1, x2, y2, s, c in dets.tolist()]
    with quiet():
        ev = COCOeval(gt, gt.loadRes(anns), "bbox")
        # IoU 0.5 and one area range only (the defaults are 10 IoUs x 4 areas): far less work and RAM on the Pi
        ev.params.iouThrs = np.array([0.5])
        ev.params.areaRng, ev.params.areaRngLbl = [[0, 1e5 ** 2]], ["all"]
        ev.params.maxDets = [max_det]
        ev.evaluate()
        ev.accumulate()
    acc = {"mAP50": ap(ev.eval["precision"]), "AR": ap(ev.eval["recall"])}
    acc.update(pr_at_iou50(ev, conf))
    return acc


def ap(a):
    a = a[a > -1]  # -1: a class without ground truth
    return round(float(a.mean()), 4) if a.size else None


def pr_at_iou50(ev, conf):
    """Precision / recall / F1 at IoU 0.5 for detections with score >= conf, plus the F1-optimal conf.
    Reuses COCOeval's greedy matching: dropping low-score detections never changes the matches of higher-score
    ones, so filtering after matching equals matching only the kept detections."""
    scores, hits, n_gt = [], [], 0
    for e in ev.evalImgs:
        if e is None:  # no ground truth and no detections for this image and class
            continue
        n_gt += int(np.sum(np.asarray(e["gtIgnore"]) == 0))
        keep = ~np.asarray(e["dtIgnore"][0], dtype=bool)
        scores.append(np.asarray(e["dtScores"])[keep])
        hits.append(np.asarray(e["dtMatches"][0])[keep] > 0)
    order = np.argsort(-np.concatenate(scores), kind="mergesort")
    s, h = np.concatenate(scores)[order], np.concatenate(hits)[order]
    tp, fp = np.cumsum(h), np.cumsum(~h)
    p, r = tp / np.maximum(tp + fp, 1), tp / max(n_gt, 1)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    k = int(np.searchsorted(-s, -conf, side="right"))  # number of detections with score >= conf
    best = int(np.argmax(f1))

    def at(a):
        return round(float(a[k - 1]), 4) if k else 0.0

    return {"P": at(p), "R": at(r), "F1": at(f1), "best_F1": round(float(f1[best]), 4),
            "best_F1_conf": round(float(s[best]), 3), "gt_boxes": n_gt}
