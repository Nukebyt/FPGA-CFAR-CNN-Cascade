# -*- coding: utf-8 -*-
"""Literature-comparable software metrics of the cascade (config A prescreen + INT8 CNN) on the CNN-study test split (842 images, 2,877 ships)
and the Weibull-only prescreen on the whole data set.  Output: Results/fixedpoint/software_metrics.json / .csv
Definitions (ship level, see the comparison report):
  recall      = ships with >= 1 accepted event within 4 px of the ship polygon / all ships (ships the prescreen never delivers count as missed)
  FA/img      = accepted events farther than 4 px from every ship, per image
  precision   = TP ships / (TP ships + false events)           (a false event is a false positive; several events on one ship count once)
  F1          = 2PR/(P+R)
Thresholds come from VALIDATION images (ship-retention targets) and are applied to the test images.
usage: HWDATA=pooldetfxA python software_metrics.py [score_name=pf_plain_q8]"""
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

name = sys.argv[1] if len(sys.argv) > 1 else "pf_plain_q8"
d = H.HWData()
m = np.load(os.path.join(H.CACHE, "meta.npz")); gt2 = m["gtIdx2"].astype(np.int64)
sv = np.load(os.path.join(H.RES, "hw", f"{name}_val.npy")).astype(np.float64)
st = np.load(os.path.join(H.RES, "hw", f"{name}_test.npy")).astype(np.float64)
names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
S = pd.read_csv(os.path.join(ROOT, "_comparison", "Results", "pd_study", "phase1_ship_table.csv"), usecols=["name", "annIdx", "scene"], low_memory=False)
S["rk"] = S.groupby("name")["annIdx"].rank(method="first").astype(int)           # annIdx is global; the per-image ship order is its rank inside the image
scene = {(r.name, int(r.rk)): r.scene for r in S.itertuples()}


def ship_scores(idx, scores):
    img, g1, g2, lab = d.img[idx], d.gt[idx], gt2[idx], d.labels[idx]
    k = np.concatenate([(img * 256 + g1)[lab & (g1 > 0)], (img * 256 + g2)[g2 > 0]]); s = np.concatenate([scores[lab & (g1 > 0)], scores[g2 > 0]])
    o = np.argsort(k, kind="stable"); k, s = k[o], s[o]
    u, f = np.unique(k, return_index=True)
    return u, np.maximum.reduceat(s, f)


ti, vi = d.idx["test"], d.idx["val"]
timgs = d.split_imgs["test"]; n_img = len(timgs)
tot_ships = int(d.nships[timgs].sum())
# scene of every ship of the test images (to count missed ships by scene)
ship_scene = {}
for im in timgs:
    for s in range(1, int(d.nships[im]) + 1):
        ship_scene[im * 256 + s] = scene.get((names[im], s), "offshore")
tot_in = sum(v == "inshore" for v in ship_scene.values()); tot_off = len(ship_scene) - tot_in
uv, smv = ship_scores(vi, sv)
u, sm = ship_scores(ti, st)
sc_of = np.array([ship_scene[int(k)] == "inshore" for k in u])
rows = []
# prescreen only (no CNN): every event accepted
n_fa_pre = int((~d.labels[ti]).sum())
tp = len(u)
rows.append(dict(system="Weibull prescreen only (integer; config A)", target="-", recall=tp / tot_ships, recall_inshore=sc_of.sum() / tot_in, recall_offshore=(~sc_of).sum() / tot_off,
                 FA_per_img=n_fa_pre / n_img, events_per_img=len(ti) / n_img, precision=tp / (tp + n_fa_pre), F1=0.0))
for r in (0.80, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
    thr = float(np.sort(smv)[int(np.floor((1 - r) * len(smv)))])
    acc = st >= thr
    fa = int((acc & ~d.labels[ti]).sum()); det = sm >= thr; tp = int(det.sum())
    P = tp / (tp + fa); R = tp / tot_ships
    rows.append(dict(system="cascade: prescreen + INT8 CNN", target=r, recall=R, recall_inshore=(det & sc_of).sum() / tot_in, recall_offshore=(det & ~sc_of).sum() / tot_off,
                     FA_per_img=fa / n_img, events_per_img=acc.sum() / n_img, precision=P, F1=2 * P * R / (P + R)))
df = pd.DataFrame(rows)
r0 = df.iloc[0]; P, R = r0.precision, r0.recall; df.loc[0, "F1"] = 2 * P * R / (P + R)
best = df.iloc[1:].F1.idxmax()
os.makedirs(os.path.join(ROOT, "_comparison", "Results", "fixedpoint"), exist_ok=True)
df.to_csv(os.path.join(ROOT, "_comparison", "Results", "fixedpoint", f"software_metrics_{name}.csv"), index=False)
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: "%.4f" % v)
print(f"test split: {n_img} images, {tot_ships} ships ({tot_in} inshore, {tot_off} offshore)")
print(df.to_string(index=False)); print("best-F1 row:", int(best))
