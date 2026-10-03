# -*- coding: utf-8 -*-
"""Numbers for the HRSID literature-comparison report -> Results/fixedpoint/comparison_data.json"""
import json, os, sys, subprocess
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "_comparison", "Results", "fixedpoint")
d = json.load(open(os.path.join(OUT, "eval.json"))); names = d["names"]
S = pd.read_csv(os.path.join(ROOT, "_comparison", "Results", "pd_study", "phase1_ship_table.csv"), usecols=["name", "annIdx", "scene"], low_memory=False)
S["rk"] = S.groupby("name")["annIdx"].rank(method="first").astype(int)
scene = {(r.name, int(r.rk)): r.scene for r in S.itertuples()}
res = {}
for ci, c in enumerate(d["configs"]):
    for m in ("float_exact", "fixed"):
        tot = {"inshore": [0, 0], "offshore": [0, 0]}; ev = 0; off = 0
        for k, r in enumerate(d["res"]):
            row = r["cfg"][ci][m]; ev += row["nEv"]; off += row["nOff"]
            for s, md in enumerate(row["md"], 1):
                sc = scene[(names[k], s)]; tot[sc][1] += 1; tot[sc][0] += md <= 4
        n = tot["inshore"][1] + tot["offshore"][1]
        res[f"{c}|{m}"] = dict(recall=(tot["inshore"][0] + tot["offshore"][0]) / n, inshore=tot["inshore"][0] / tot["inshore"][1], offshore=tot["offshore"][0] / tot["offshore"][1],
                               ships=n, n_inshore=tot["inshore"][1], n_offshore=tot["offshore"][1], events_per_img=ev / len(names), offship_per_img=off / len(names))
json.dump(res, open(os.path.join(OUT, "comparison_data.json"), "w"), indent=1)
for k, v in res.items(): print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
