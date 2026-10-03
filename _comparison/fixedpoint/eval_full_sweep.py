# -*- coding: utf-8 -*-
"""Whole-data-set evaluation of the ON-BOARD sweep (cascade_ctx_jtag / cascade_ps_jtag, Results/<sweep>/frame_*.npz: events j, i, integer logit, accept).
One sweep gives both systems: Weibull prescreen only = every event accepted; prescreen + CNN = events with logit >= theta.
Ground truth: an event is ON a ship if it lies within 4 full-resolution px of the ship polygon (events at (2j+1, 2i+1)); ship recall counts ships with >= 1 accepted
on-ship event; false events are accepted events farther than 4 px from every ship.  Subsets: all 5,604 images, train / val / test (the CNN split; train+val are
in-sample for the CNN, the prescreen has no training), inshore / offshore ships.  Image-level bootstrap (1,000 resamples) gives 95 % confidence intervals.
usage: python eval_full_sweep.py <sweep_dir_name> <tag> [workers=2]      -> Results/fixedpoint/full_<tag>_*.csv|json, figure"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prescreen_fx as fx
from eval_fx_dataset import load_ann, ship_masks
from scipy import ndimage

ROOT = fx.ROOT
RES = os.path.join(ROOT, "_comparison", "Results")
sweep, tag = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TOL = 4
THETAS = {"90": -5979, "95": -10695, "97": -14462, "98": -17056}          # validation thresholds of the context network (override with --theta-json)
if len(sys.argv) > 4:
    THETAS = json.load(open(sys.argv[4]))
names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
S = pd.read_csv(os.path.join(RES, "pd_study", "phase1_ship_table.csv"), usecols=["name", "annIdx", "scene"], low_memory=False)
S["rk"] = S.groupby("name")["annIdx"].rank(method="first").astype(int)
scene = {(r.name, int(r.rk)): r.scene for r in S.itertuples()}
by = load_ann()


def work(k):
    nm = names[k]
    f = np.load(os.path.join(RES, sweep, f"frame_{k:04d}.npz"), allow_pickle=True)
    j, i, lg = f["j"].astype(np.int64), f["i"].astype(np.int64), f["logit"].astype(np.int64)
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", nm)))
    h0, w0 = (im.shape[0] // 2) * 2, (im.shape[1] // 2) * 2
    masks = ship_masks(by.get(nm, []), h0, w0)
    dists = [ndimage.distance_transform_edt(~m) for m in masks]
    ry, rx = 2 * j + 1, 2 * i + 1
    if dists:
        dm = np.stack([d[ry, rx] for d in dists], 0) if len(j) else np.zeros((len(dists), 0))
        dmin = dm.min(0) if len(j) else np.zeros(0)
        ship_score = np.array([lg[dm[s] <= TOL].max() if (dm[s] <= TOL).any() else -(2 ** 40) for s in range(len(dists))])
    else:
        dmin = np.full(len(j), np.inf); ship_score = np.zeros(0)
    sc = np.array([scene[(nm, s + 1)] == "inshore" for s in range(len(masks))])
    return k, lg.astype(np.int64), (dmin > TOL), ship_score, sc


if __name__ == "__main__":
    t0 = time.time(); res = [None] * len(names)
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(work, range(len(names)), chunksize=8)):
            res[r[0]] = r
            if (n + 1) % 500 == 0:
                print("[%d/%d] %.0fs" % (n + 1, len(names), time.time() - t0), flush=True)
    n_img = len(names)
    order = np.arange(n_img); np.random.RandomState(20260925).shuffle(order)
    ntr, nva = int(0.7 * n_img), int(0.15 * n_img)
    subsets = {"all": np.arange(n_img), "train": np.sort(order[:ntr]), "val": np.sort(order[ntr:ntr + nva]), "test": np.sort(order[ntr + nva:])}

    def stats(theta):
        """per-image sufficient statistics: tp, tp_inshore, tp_offshore, ships, n_inshore, n_offshore, false events, accepted events"""
        out = np.zeros((n_img, 8))
        for k in range(n_img):
            _, lg, off, ss, sc = res[k]
            a = lg >= theta; det = ss >= theta
            out[k] = [det.sum(), (det & sc).sum(), (det & ~sc).sum(), len(ss), sc.sum(), (~sc).sum(), (a & off).sum(), a.sum()]
        return out

    def summarize(st, idx):
        t = st[idx].sum(0)
        R = t[0] / t[3]; fa = t[6]; P = t[0] / max(t[0] + fa, 1)
        return dict(recall=R, recall_inshore=t[1] / max(t[4], 1), recall_offshore=t[2] / max(t[5], 1), FA_per_img=fa / len(idx), accepted_per_img=t[7] / len(idx), precision=P,
                    F1=2 * P * R / max(P + R, 1e-12), ships=int(t[3]))

    systems = {"Weibull only (prescreen)": -(2 ** 40) + 1}
    systems.update({f"cascade @ {k} % val retention (theta {v})": v for k, v in THETAS.items()})
    ST = {n: stats(t) for n, t in systems.items()}
    rng = np.random.RandomState(7); rows = []
    for sname, idx in subsets.items():
        boots = [rng.choice(idx, len(idx), replace=True) for _ in range(1000)]
        for sysname in systems:
            m = summarize(ST[sysname], idx)
            bs = pd.DataFrame([summarize(ST[sysname], b) for b in boots])
            row = dict(subset=sname, system=sysname, **m)
            for k in ("recall", "FA_per_img", "precision", "F1", "recall_inshore"):
                row[k + "_lo"], row[k + "_hi"] = np.percentile(bs[k], 2.5), np.percentile(bs[k], 97.5)
            rows.append(row)
        print("subset", sname, "done %.0fs" % (time.time() - t0), flush=True)
    ths = np.linspace(-30000, 4000, 69)
    curves = {sname: [dict(theta=float(t), **summarize(stats(t), subsets[sname])) for t in ths] for sname in ("all", "test")}
    json.dump(curves, open(os.path.join(RES, "fixedpoint", f"full_{tag}_curves.json"), "w"))
    df = pd.DataFrame(rows)
    out = os.path.join(RES, "fixedpoint"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"full_{tag}_metrics.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: "%.4f" % v)
    print(df[["subset", "system", "recall", "recall_lo", "recall_hi", "recall_inshore", "FA_per_img", "FA_per_img_lo", "FA_per_img_hi", "precision", "F1"]].to_string(index=False))
