# -*- coding: utf-8 -*-
"""Aggregate the per-run bootstrap results (seeds_metrics.py) into the confidence-interval table: seeds (same split), splits (seed 1), pooled mean +- s.d.
Writes Results/fixedpoint/seeds_summary.csv and prints the table."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "_comparison", "Results", "fixedpoint", "seeds")
RUNS = [  # tag, split seed, ctx score name, plain score name
    ("sd1", "20260925", "pf_full_q8", "pf_plain_q8"), ("sd2", "20260925", "ctx_q_sd2", "pl_q_sd2"), ("sd3", "20260925", "ctx_q_sd3", "pl_q_sd3"),
    ("sp2", "20261004", "ctx_q_sp2", "pl_q_sp2"), ("sp3", "20261005", "ctx_q_sp3", "pl_q_sp3")]
rows = []
for tag, split, cn, pn in RUNS:
    for kind, nm in (("context", cn), ("single-tower", pn)):
        f = os.path.join(OUT, f"{tag}_{nm}.json")
        if not os.path.exists(f):
            if os.path.exists(os.path.join(ROOT, "_comparison", "Results", "hw", f"{nm}_test.npy")):
                subprocess.run([sys.executable, os.path.join(HERE, "seeds_metrics.py"), nm, tag], env=dict(os.environ, HWSPLIT=split, HWDATA="pooldetfxA"), check=True)
            else:
                continue
        j = json.load(open(f))
        for r, v in j["results"].items():
            rows.append(dict(model=kind, run=tag, target=float(r), n_test_images=j["n_test_images"], **{k: v[k] for k in v}))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "_comparison", "Results", "fixedpoint", "seeds_all_runs.csv"), index=False)
summ = []
for kind in df.model.unique():
    for tgt in sorted(df.target.unique()):
        d = df[(df.model == kind) & (df.target == tgt)]
        seeds = d[d.run.isin(["sd1", "sd2", "sd3"])]
        row = dict(model=kind, target=tgt, n_runs=len(d), n_seeds=len(seeds))
        for k in ("recall", "recall_inshore", "FA_per_img", "precision", "F1"):
            row[k + "_mean"] = d[k].mean(); row[k + "_sd"] = d[k].std(ddof=1) if len(d) > 1 else np.nan
            row[k + "_seedsd"] = seeds[k].std(ddof=1) if len(seeds) > 1 else np.nan
        main = d[d.run == "sd1"]
        if len(main):
            for k in ("recall", "FA_per_img", "F1"):
                row[k + "_boot_lo"], row[k + "_boot_hi"] = float(main[k + "_lo"].iloc[0]), float(main[k + "_hi"].iloc[0])
        summ.append(row)
S = pd.DataFrame(summ)
S.to_csv(os.path.join(ROOT, "_comparison", "Results", "fixedpoint", "seeds_summary.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: "%.4f" % v)
print(S[["model", "target", "n_runs", "recall_mean", "recall_sd", "FA_per_img_mean", "FA_per_img_sd", "F1_mean", "recall_boot_lo", "recall_boot_hi", "FA_per_img_boot_lo", "FA_per_img_boot_hi"]].to_string(index=False))
