# -*- coding: utf-8 -*-
"""Test-split metrics of one trained INT8 model at validation-calibrated thresholds, with image-level bootstrap 95 % CIs.
usage: HWSPLIT=<split seed> HWDATA=pooldetfxA python seeds_metrics.py <score_name> <tag>      -> Results/fixedpoint/seeds/<tag>_<score_name>.json
(the split seed must be the one the model was trained with)"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "_comparison", "cnn"))
os.environ.setdefault("HWDATA", "pooldetfxA")
import hwlib as H

name, tag = sys.argv[1], sys.argv[2]
d = H.HWData()
m = np.load(os.path.join(H.CACHE, "meta.npz")); gt2 = m["gtIdx2"].astype(np.int64)
sv = np.load(os.path.join(H.RES, "hw", f"{name}_val.npy")).astype(np.float64)
st = np.load(os.path.join(H.RES, "hw", f"{name}_test.npy")).astype(np.float64)
names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
S = pd.read_csv(os.path.join(ROOT, "_comparison", "Results", "pd_study", "phase1_ship_table.csv"), usecols=["name", "annIdx", "scene"], low_memory=False)
S["rk"] = S.groupby("name")["annIdx"].rank(method="first").astype(int)
scene = {(r.name, int(r.rk)): r.scene for r in S.itertuples()}


def ship_scores(idx, scores):
    img, g1, g2, lab = d.img[idx], d.gt[idx], gt2[idx], d.labels[idx]
    k = np.concatenate([(img * 256 + g1)[lab & (g1 > 0)], (img * 256 + g2)[g2 > 0]]); s = np.concatenate([scores[lab & (g1 > 0)], scores[g2 > 0]])
    o = np.argsort(k, kind="stable"); k, s = k[o], s[o]
    u, f = np.unique(k, return_index=True)
    return u, np.maximum.reduceat(s, f)


ti, vi = d.idx["test"], d.idx["val"]
timgs = d.split_imgs["test"]; pos_of = {int(im): n for n, im in enumerate(timgs)}
ship_info = []                                               # (image position in the test list, inshore flag) of every test ship
for im in timgs:
    for s in range(1, int(d.nships[im]) + 1):
        ship_info.append((pos_of[int(im)], scene.get((names[im], s), "offshore") == "inshore", int(im) * 256 + s))
sk = {k: n for n, (_, _, k) in enumerate(ship_info)}
u, sm = ship_scores(ti, st)
best = np.full(len(ship_info), -np.inf)
for kk, v in zip(u, sm):
    best[sk[int(kk)]] = v
uv, smv = ship_scores(vi, sv)
img_pos = np.array([pos_of[int(x)] for x in d.img[ti]])
off = ~d.labels[ti]
sc = np.array([x[1] for x in ship_info]); spos = np.array([x[0] for x in ship_info])
n_img = len(timgs); rng = np.random.RandomState(3)
res = {}
for r in (0.90, 0.95, 0.97, 0.98):
    thr = float(np.sort(smv)[int(np.floor((1 - r) * len(smv)))])
    a = st >= thr
    det = best >= thr
    # per-image sufficient statistics
    tp = np.bincount(spos, weights=det, minlength=n_img); tpi = np.bincount(spos, weights=det & sc, minlength=n_img)
    ships = np.bincount(spos, minlength=n_img).astype(float); shi = np.bincount(spos, weights=sc, minlength=n_img)
    fa = np.bincount(img_pos, weights=a & off, minlength=n_img); acc = np.bincount(img_pos, weights=a, minlength=n_img)
    def summ(idx):
        T, TI, SH, SHI, FA, AC = tp[idx].sum(), tpi[idx].sum(), ships[idx].sum(), shi[idx].sum(), fa[idx].sum(), acc[idx].sum()
        R = T / SH; P = T / max(T + FA, 1)
        return dict(recall=R, recall_inshore=TI / max(SHI, 1), FA_per_img=FA / len(idx), precision=P, F1=2 * P * R / max(P + R, 1e-12), accepted_per_img=AC / len(idx))
    base = summ(np.arange(n_img))
    bs = pd.DataFrame([summ(rng.choice(n_img, n_img, replace=True)) for _ in range(1000)])
    res[str(r)] = dict(base, **{k + "_lo": float(np.percentile(bs[k], 2.5)) for k in bs.columns}, **{k + "_hi": float(np.percentile(bs[k], 97.5)) for k in bs.columns}, theta_real=thr)
out = os.path.join(ROOT, "_comparison", "Results", "fixedpoint", "seeds"); os.makedirs(out, exist_ok=True)
json.dump(dict(name=name, tag=tag, n_test_images=n_img, n_ships=len(ship_info), results=res), open(os.path.join(out, f"{tag}_{name}.json"), "w"), indent=1)
print(name, tag, {k: (round(v["recall"], 4), round(v["FA_per_img"], 2)) for k, v in res.items()})
