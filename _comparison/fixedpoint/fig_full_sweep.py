# -*- coding: utf-8 -*-
"""Recall vs false events per image from the on-board whole-data-set sweep (curves JSON of eval_full_sweep.py) -> Figures/fixedpoint_full_sweep.png"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "Results", "fixedpoint")
C = json.load(open(os.path.join(RES, "full_ctx_curves.json"))); M = pd.read_csv(os.path.join(RES, "full_ctx_metrics.csv"))
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, (sub, ttl) in zip(ax, (("all", "All 5,604 images (train+val in-sample for the CNN)"), ("test", "Held-out test split (842 images)"))):
    c = [r for r in C[sub] if r["FA_per_img"] <= 12]
    a.plot([r["FA_per_img"] for r in c], [r["recall"] for r in c], "-", color="C0", label="Weibull + CNN (on-board, all thetas)")
    a.plot([r["FA_per_img"] for r in c], [r["recall_inshore"] for r in c], "--", color="C3", label="  inshore ships only")
    m = M[(M.subset == sub) & M.system.str.startswith("cascade")]
    a.errorbar(m.FA_per_img, m.recall, yerr=[m.recall - m.recall_lo, m.recall_hi - m.recall], fmt="o", color="k", capsize=3, label="validation-calibrated operating points (95% CI)")
    w = M[(M.subset == sub) & M.system.str.startswith("Weibull")].iloc[0]
    a.annotate("Weibull only: recall %.3f at %.0f false events/img" % (w.recall, w.FA_per_img), (0.98, 0.45), xycoords="axes fraction", ha="right", fontsize=8)
    a.set_title(ttl, fontsize=9); a.set_xlabel("false events per image"); a.grid(alpha=.3); a.set_ylim(0.7, 1.0)
ax[0].set_ylabel("ship recall"); ax[0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); out = os.path.join(HERE, "..", "Figures", "fixedpoint_full_sweep.png"); plt.savefig(out, dpi=170); print(out)
