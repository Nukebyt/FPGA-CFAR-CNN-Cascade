# -*- coding: utf-8 -*-
"""Paired comparison context vs single tower over the runs of seeds_all_runs.csv (same split and seed in each pair) -> Results/fixedpoint/seeds_paired.csv"""
import os
import pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "fixedpoint")
d = pd.read_csv(os.path.join(R, "seeds_all_runs.csv"))
c = d[d.model == "context"].set_index(["run", "target"]); s = d[d.model == "single-tower"].set_index(["run", "target"])
rows = []
for t in sorted(d.target.unique()):
    runs = [r for r in c.index.get_level_values(0).unique() if (r, t) in s.index]
    dR = [c.loc[(r, t), "recall"] - s.loc[(r, t), "recall"] for r in runs]
    dF = [c.loc[(r, t), "FA_per_img"] - s.loc[(r, t), "FA_per_img"] for r in runs]
    rF = [1 - c.loc[(r, t), "FA_per_img"] / s.loc[(r, t), "FA_per_img"] for r in runs]
    dI = [c.loc[(r, t), "recall_inshore"] - s.loc[(r, t), "recall_inshore"] for r in runs]
    rows.append(dict(target=t, n_pairs=len(runs), dRecall_mean=sum(dR) / len(dR), dRecall_min=min(dR), dRecall_max=max(dR), dInshore_mean=sum(dI) / len(dI),
                     FA_reduction_mean=sum(rF) / len(rF), FA_reduction_min=min(rF), FA_reduction_max=max(rF), ctx_fewer_FA_in=sum(x < 0 for x in dF)))
o = pd.DataFrame(rows); o.to_csv(os.path.join(R, "seeds_paired.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: "%.4f" % v); print(o.to_string(index=False))
