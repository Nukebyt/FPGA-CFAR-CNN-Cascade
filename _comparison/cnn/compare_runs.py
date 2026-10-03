# -*- coding: utf-8 -*-
"""Compare saved-score runs with a smoother statistic than a single retention
point: mean FA/img over ship retention 0.70..0.90, plus image-bootstrap CIs."""
import sys, os, json, glob
import numpy as np
os.environ.setdefault("HWDATA", "64"); os.environ.setdefault("HWGATE", "0.45")
import hwlib as H

d = H.HWData()
GRID = (0.70, 0.75, 0.80, 0.85, 0.90)


def summary(split, scores, img_sel=None):
    idx = d.idx[split]
    if img_sel is not None:
        m = np.isin(d.img[idx], img_sel)
        idx, scores = idx[m], scores[m]
    c = H.cascade_curve(d, idx, scores, rets=GRID + (0.95,))
    return np.mean([c[str(r)]["fa_per_img"] for r in GRID]), c


def boot(split, scores, B=200, seed=0):
    rng = np.random.RandomState(seed)
    imgs = d.split_imgs[split]
    vals = []
    for _ in range(B):
        sel = rng.choice(imgs, len(imgs), replace=True)
        # resample images with replacement via weights: build an index list
        idx = d.idx[split]
        cnt = np.bincount(d.img[idx], minlength=len(d.nships))
        w = np.bincount(sel, minlength=len(d.nships))[d.img[idx]]
        rep = np.repeat(np.arange(len(idx)), w)
        sub = d.idx[split][rep]
        # recompute with replicated clusters (ships replicate too through their clusters)
        pos = d.labels[sub]
        key = (d.img[sub] * 256 + d.gt[sub])[pos]
        # treat each replicate of an image as a distinct image for ship keys
        rep_img = np.repeat(d.img[idx], w)
        occ = np.cumsum(np.r_[0, (np.diff(rep_img) != 0).astype(int)])  # new id whenever image run changes
        key = (occ[pos] * 256 + d.gt[sub][pos])
        order = np.argsort(key, kind="stable"); ks = key[order]
        u, st = np.unique(ks, return_index=True)
        sp = scores[rep][pos][order]
        sm = np.sort(np.maximum.reduceat(sp, st))
        neg = np.sort(scores[rep][~pos])
        v = []
        for r in GRID:
            t = sm[int(np.floor((1 - r) * len(sm)))]
            v.append((len(neg) - np.searchsorted(neg, t, side="left")) / len(sel))
        vals.append(np.mean(v))
    return np.percentile(vals, [2.5, 97.5])


names = sys.argv[1:] or sorted({os.path.basename(f)[:-9] for f in glob.glob(H.RES + r"\hw\*_test.npy") if "_int_" not in f})
rows = []
for n in names:
    try:
        sv = np.load(H.RES + rf"\hw\{n}_val.npy").astype(np.float64)
        st = np.load(H.RES + rf"\hw\{n}_test.npy").astype(np.float64)
    except Exception:
        continue
    if len(sv) != len(d.idx['val']):
        continue   # scores from the other dataset variant
    mv, _ = summary("val", sv)
    mt, ct = summary("test", st)
    lo, hi = boot("test", st, B=100)
    rows.append((n, mv, mt, lo, hi, ct))
rows.sort(key=lambda r: r[1])
print(f"{'run':14s} {'val mean FA':>11s} {'test mean FA':>12s} {'[95% CI]':>14s} | test FA@80 @85 @90 @95")
for n, mv, mt, lo, hi, ct in rows:
    print(f"{n:14s} {mv:11.2f} {mt:12.2f} [{lo:5.2f},{hi:5.2f}] | " +
          " ".join(f"{ct[k]['fa_per_img']:5.1f}" for k in ('0.8', '0.85', '0.9', '0.95')))
