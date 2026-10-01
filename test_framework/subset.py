"""A smaller VisDrone test set with the same class ratio as the full one; accuracy.py uses it by default:

    python test_framework/subset.py          # input/VisDrone2019-DET-test-dev -> input/VisDrone2019-DET-test-dev-500

Picks -n images and hard-links them with their annotations into a new folder (no extra disk space; the full set
stays as it is). Boxes are counted after the training filtering (dataset.py). Every image group (a 9999xxx set, or
the single frames of the video clips) gives n/total of its images, first spread evenly over it, so the scenes are
mixed as in the full set. Then images are swapped within their group until every class has as close as possible to
n/total of its boxes in the full set: the class ratio and the boxes per image stay the same. There is no randomness:
the same full set always gives the same images.
"""
import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(REPO))  # common.py lives in the repo root
from common import VISDRONE
from dataset import image_files, read_boxes


def image_groups(files):
    """9999938_00000_d_0000100.jpg -> 9999938_00000_d; the images that are alone in their group (frames of
    different video clips) share one group."""
    keys = [f.rsplit("_", 1)[0] for f in files]
    size = Counter(keys)
    return np.array([k if size[k] > 1 else "single" for k in keys])


def allocate(groups, n):
    """Images to take from every group: n/total of it, rounded by largest remainder so that they sum to n."""
    exact = {g: s * n / len(groups) for g, s in Counter(groups).items()}
    take = {g: int(e) for g, e in exact.items()}
    for g in sorted(exact, key=lambda g: exact[g] - take[g], reverse=True)[:n - sum(take.values())]:
        take[g] += 1
    return take


def select(counts, groups, n):
    """counts: boxes per class of every image, (images, classes). Returns the indices of the n chosen images."""
    chosen = np.zeros(len(counts), bool)
    for g, k in allocate(groups, n).items():  # start: evenly spread over every group
        idx = np.flatnonzero(groups == g)
        chosen[idx[np.arange(k) * len(idx) // k]] = True
    target = counts.sum(0) * n / len(counts)

    def cost(total):  # relative error, so that a rare class counts as much as cars
        return (((total - target) / target) ** 2).sum(-1)

    total = counts[chosen].sum(0)
    while True:  # best swap of one chosen image for one other image of its group, until none helps
        best, swap = cost(total), None
        for g in np.unique(groups):
            ins, outs = np.flatnonzero(chosen & (groups == g)), np.flatnonzero(~chosen & (groups == g))
            if len(ins) and len(outs):
                c = cost(total + counts[outs][None] - counts[ins][:, None])  # (ins, outs)
                i, o = np.unravel_index(c.argmin(), c.shape)
                if c[i, o] < best:
                    best, swap = c[i, o], (ins[i], outs[o])
        if swap is None:
            return np.flatnonzero(chosen)
        chosen[swap[0]], chosen[swap[1]] = False, True
        total = counts[chosen].sum(0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(REPO / "input" / "VisDrone2019-DET-test-dev"),
                   help="full VisDrone folder with images/ and annotations/")
    p.add_argument("-n", type=int, default=500, help="images to keep (default: 500)")
    p.add_argument("--out", help="new folder (default: DATA-N)")
    a = p.parse_args()
    data = a.data.rstrip("/")
    out = a.out or f"{data}-{a.n}"
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists: delete it first")
    files = image_files(data)
    if not 0 < a.n < len(files):
        raise SystemExit(f"-n must be between 1 and {len(files) - 1}")
    counts = np.zeros((len(files), len(VISDRONE)), int)
    for i, fname in enumerate(files):
        for cat, _ in read_boxes(data, fname):
            counts[i, cat] += 1
    keep = select(counts, image_groups(files), a.n)

    for d in ("images", "annotations"):
        os.makedirs(os.path.join(out, d))
    for i in keep:
        for d, fname in (("images", files[i]), ("annotations", os.path.splitext(files[i])[0] + ".txt")):
            src, dst = os.path.join(data, d, fname), os.path.join(out, d, fname)
            try:
                os.link(src, dst)
            except OSError:  # another file system
                shutil.copy2(src, dst)

    full, sub = counts.sum(0), counts[keep].sum(0)
    print(f"{'class':<16}{'full':>7}{'%':>7}{'subset':>8}{'%':>7}")
    for name, f, s in zip(VISDRONE, full, sub):
        print(f"{name:<16}{f:>7}{100 * f / full.sum():>7.2f}{s:>8}{100 * s / sub.sum():>7.2f}")
    print(f"{len(keep)} of {len(files)} images, {sub.sum()} of {full.sum()} boxes ({sub.sum() / len(keep):.1f} "
          f"per image, full set {full.sum() / len(files):.1f}) -> {out}")


if __name__ == "__main__":
    main()
