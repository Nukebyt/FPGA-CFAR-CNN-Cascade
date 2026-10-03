# -*- coding: utf-8 -*-
"""Summarise Results/fixedpoint/eval.json: float vs 8-bit-data vs integer prescreen, whole data set and the CNN-study test split."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "_comparison", "Results", "fixedpoint")
d = json.load(open(os.path.join(OUT, "eval.json")))
cfgs, res, names = d["configs"], d["res"], d["names"]
n_img = len(res)
order = np.arange(n_img); np.random.RandomState(20260925).shuffle(order)
ntr, nva = int(0.7 * n_img), int(0.15 * n_img)
split = {"all": np.arange(n_img), "test": np.sort(order[ntr + nva:])}
rows = []
for ci, c in enumerate(cfgs):
    for sname, idx in split.items():
        for m in ("float_exact", "ideal_q8", "fixed"):
            md = np.concatenate([np.array(res[k]["cfg"][ci][m]["md"]) for k in idx if res[k]["nShips"]])
            ev = np.mean([res[k]["cfg"][ci][m]["nEv"] for k in idx]); off = np.mean([res[k]["cfg"][ci][m]["nOff"] for k in idx])
            full = np.mean([all(x <= 4 for x in res[k]["cfg"][ci][m]["md"]) for k in idx if res[k]["nShips"]])
            rows.append(dict(config=c, subset=sname, arithmetic=m, ships=len(md), Pd=(md <= 4).mean(), missed=int((md > 4).sum()), full_delivery=full,
                             events_per_img=ev, offship_events_per_img=off))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "fx_vs_float_pd.csv"), index=False)
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: "%.4f" % v)
print(df.to_string(index=False))
# decision agreement
tot_px = 0
agree = []
for ci, c in enumerate(cfgs):
    npx = sum(400 * 400 for _ in res)
    a = dict(config=c, px_total=npx,
             mis_fixed_vs_ideal=sum(r["cfg"][ci]["mis_fixed_vs_ideal"] for r in res), mis_ideal_vs_float=sum(r["cfg"][ci]["mis_ideal_vs_float"] for r in res),
             mis_fixed_vs_float=sum(r["cfg"][ci]["mis_fixed_vs_float"] for r in res), event_set_diff_fixed_vs_ideal=sum(r["cfg"][ci]["evsym_fixed_vs_ideal"] for r in res),
             det_px_float=sum(r["cfg"][ci]["float_exact"]["nDet"] for r in res))
    agree.append(a)
ag = pd.DataFrame(agree)
ag["mis_fixed_vs_ideal_ppm"] = ag.mis_fixed_vs_ideal / ag.px_total * 1e6
ag["mis_ideal_vs_float_pct_of_det"] = ag.mis_ideal_vs_float / ag.det_px_float * 100
ag.to_csv(os.path.join(OUT, "fx_decision_agreement.csv"), index=False)
print(ag.to_string(index=False))
