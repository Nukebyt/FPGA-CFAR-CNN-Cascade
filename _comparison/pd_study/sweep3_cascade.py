# -*- coding: utf-8 -*-
"""After-CNN (cascade) results for the pooled prescreen at Pfa 1e-2, 1e-3, 1e-4, plus the Weibull-only numbers side by side.
Models: cnn/train_ctx.py (context tower + side features + ship-level loss + hard-negative mining), one per Pfa  (run names pcC2/pcC3/pcC4_s1).
The CNN threshold is picked on the VALIDATION images for a ship-retention target and applied to every image.  Train images are in-sample for the CNN:
all headline numbers are on the held-out test split; "all images" is shown for completeness.
usage: python sweep3_cascade.py [--target 0.97]       Outputs: Results/pd_sweep3/report/"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "_comparison", "Results")
OUT = os.path.join(RES, "pd_sweep3", "report")
ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.97); a = ap.parse_args()
PFL = ["1e-2", "1e-3", "1e-4"]; RUN = {"1e-2": ("pooldetC2", "pcC2_s1"), "1e-3": ("pooldetC3", "pcC3_s1"), "1e-4": ("pooldetC4", "pcC4_s1")}
TARGETS = [0.80, 0.90, 0.95, 0.97, 0.98, 0.99]
FLOOR = 1e-8
sys.path.insert(0, os.path.join(ROOT, "_comparison", "cnn"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"1e-2": "#1f4e9c", "1e-3": "#4a86d0", "1e-4": "#9cc0ea"}
wb = pd.read_csv(os.path.join(OUT, "per_image_weibull_only.csv"))
summ_wb = pd.read_csv(os.path.join(OUT, "summary_weibull_only.csv"))


def wilson(k, n):
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(c - h, 0), min(c + h, 1)


def run_one(pl):
    """returns dict with per-image arrays (indexed by 0-based image) and a table of operating points"""
    ds, name = RUN[pl]
    code = f"""
import os, sys, json, numpy as np
os.environ['HWDATA']='{ds}'
import hwlib as H
d = H.HWData(); m = np.load(os.path.join(H.CACHE, 'meta.npz')); gt2 = m['gtIdx2'].astype(np.int64)
sv = np.load(os.path.join(H.RES, 'hw', '{name}_val.npy')).astype(float); sall = np.load(os.path.join(H.RES, 'hw', '{name}_all.npy')).astype(float)
n_img = len(d.nships)
def keyed(idx, sc):
    img, g1, g2, lab = d.img[idx], d.gt[idx], gt2[idx], d.labels[idx]
    k = np.concatenate([(img * 256 + g1)[lab & (g1 > 0)], (img * 256 + g2)[g2 > 0]]); s = np.concatenate([sc[lab & (g1 > 0)], sc[g2 > 0]])
    o = np.argsort(k, kind='stable'); k, s = k[o], s[o]; u, f = np.unique(k, return_index=True); return u, np.maximum.reduceat(s, f)
vi = d.idx['val']; uv, smv = keyed(vi, sv); smv = np.sort(smv)
all_idx = np.arange(d.n); ua, sma = keyed(all_idx, sall)       # per-ship best score over ALL events
ship_img = (ua // 256).astype(int)
out = dict(n_img=n_img, ships_img=d.nships.tolist(), split_test=[int(i) for i in d.split_imgs['test']], split_val=[int(i) for i in d.split_imgs['val']], ops={{}})
neg = ~d.labels
for r in {TARGETS}:
    thr = float(smv[int(np.floor((1 - r) * len(smv)))])
    kept = np.bincount(ship_img[sma >= thr], minlength=n_img)
    fa = np.bincount(d.img[neg & (sall >= thr)], minlength=n_img)
    acc = np.bincount(d.img[sall >= thr], minlength=n_img)
    out['ops'][str(r)] = dict(thr=thr, kept=kept.tolist(), fa=fa.tolist(), acc=acc.tolist())
out['delivered'] = np.bincount(ship_img, minlength=n_img).tolist()
print('JSON' + json.dumps(out))
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, "_comparison", "cnn"), capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        raise RuntimeError(r.stderr[-1500:])
    return json.loads(line[0][4:])


res = {}
for pl in PFL:
    if not os.path.exists(os.path.join(RES, "hw", f"{RUN[pl][1]}_all.npy")):
        print("missing model for Pfa", pl); continue
    res[pl] = run_one(pl)
assert res, "no trained models found"

split = wb.set_index("img")["split"]; ships_i = wb.set_index("img")["ships"]
rows = []; per_img = wb.copy()
for pl, R in res.items():
    imgs0 = np.arange(R["n_img"])                                     # 0-based image index; wb.img is 1-based
    hold = np.array([split[i + 1] in ("val", "test") for i in imgs0]); test = np.array([split[i + 1] == "test" for i in imgs0])
    ships = np.array(R["ships_img"], float)
    for r_, op in R["ops"].items():
        kept, fa, acc = np.array(op["kept"], float), np.array(op["fa"], float), np.array(op["acc"], float)
        for lab, mk in (("test", test), ("held-out (val+test)", hold), ("all images (train in-sample)", np.ones_like(test))):
            rows.append(dict(pfa=pl, retention_target=float(r_), subset=lab, images=int(mk.sum()), ships=int(ships[mk].sum()), cascade_pd=kept[mk].sum() / ships[mk].sum(),
                             delivered_pd=np.array(R["delivered"], float)[mk].sum() / ships[mk].sum(), fa_per_img=fa[mk].mean(), accepted_per_img=acc[mk].mean(), thr=op["thr"]))
        if abs(float(r_) - a.target) < 1e-9:
            per_img[f"cascadePd_{pl}"] = kept[wb.img.values - 1] / ships[wb.img.values - 1]
            per_img[f"cascadeFAevents_{pl}"] = fa[wb.img.values - 1]
cas = pd.DataFrame(rows); cas.to_csv(os.path.join(OUT, "cascade_operating_points.csv"), index=False, float_format="%.5g")
pd.set_option("display.width", 250)
print(cas[(cas.subset == "test")].round(4).to_string(index=False))
# event Pfa per image = false accepted events / background pixels (pixels more than 4 px from every ship)
bg_img = wb["bgPix"].values
for pl in res:
    if f"cascadeFAevents_{pl}" in per_img:
        per_img[f"cascadeEventPfa_{pl}"] = per_img[f"cascadeFAevents_{pl}"] / bg_img
per_img.to_csv(os.path.join(OUT, "per_image_weibull_only_and_cascade.csv"), index=False, float_format="%.6g")

# ------------------------------------------------------------------ figures
# (1) per-image cascade results (same x-ordering as the Weibull-only per-image figure)
fig, axs = plt.subplots(len(res), 3, figsize=(16, 3.7 * len(res)), squeeze=False)
for r_, pl in enumerate(res):
    order = np.argsort(per_img[f"pixelPfa_pooled_{pl}"].values); xr = np.arange(len(per_img)); held = per_img.split.isin(["val", "test"]).values[order]
    for c_, (key, lab, logy, floor) in enumerate([(f"cascadePd_{pl}", f"cascade Pd per image (target {a.target:.2f})", False, 0), (f"cascadeFAevents_{pl}", "false events per image after CNN (log)", True, 0.5),
                                                  (f"cascadeEventPfa_{pl}", "event Pfa per image after CNN (log)", True, FLOOR)]):
        y = per_img[key].values[order]; yy = np.maximum(y, floor) if logy else y
        axs[r_, c_].plot(xr[~held], yy[~held], ".", ms=1.4, color="#c8c8c8", alpha=.6, label="train images (in-sample)")
        axs[r_, c_].plot(xr[held], yy[held], ".", ms=1.6, color=COL[pl], alpha=.7, label="held-out images")
        k = np.ones(201); sm = np.convolve(y[held] if False else y, k, "same") / np.convolve(np.ones_like(y), k, "same")
        axs[r_, c_].plot(xr, np.maximum(sm, floor) if logy else sm, "k-", lw=1.6, label="rolling mean (201)")
        if logy: axs[r_, c_].set_yscale("log")
        axs[r_, c_].set_ylabel(f"Pfa {pl}: {lab}", fontsize=8); axs[r_, c_].grid(alpha=.3, which="both")
        if r_ == len(res) - 1: axs[r_, c_].set_xlabel("image rank (sorted by the Weibull-only pixel Pfa of this setting)")
axs[0, 0].legend(fontsize=7, loc="lower left")
fig.suptitle(f"Per-image results AFTER the CNN (retention target {a.target:.0%} on validation images), pooled prescreen", fontsize=10); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_per_image_cascade_3pfa.png")); plt.close(fig)

# (2) Weibull-only vs after CNN, test split
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3)); x = np.arange(len(res)); w = .36
t = cas[(cas.subset == "test") & (np.isclose(cas.retention_target, a.target))].set_index("pfa")
pdw = [summ_wb[(summ_wb.design.str.startswith("pooled")) & (np.isclose(summ_wb.pfa, float(pl)))].pd_test.iloc[0] for pl in res]
evw = [summ_wb[(summ_wb.design.str.startswith("pooled")) & (np.isclose(summ_wb.pfa, float(pl)))].off_events_mean.iloc[0] for pl in res]
ax[0].bar(x - w / 2, pdw, w, color="#9aa7b8", edgecolor="k", lw=.5, label="Weibull only"); ax[0].bar(x + w / 2, [t.loc[pl, "cascade_pd"] for pl in res], w, color="#2e8b57", edgecolor="k", lw=.5, label="after CNN")
for xi, v in zip(x - w / 2, pdw): ax[0].text(xi, v + .004, f"{v:.3f}", ha="center", fontsize=8)
for xi, pl in zip(x + w / 2, res): ax[0].text(xi, t.loc[pl, "cascade_pd"] + .004, f"{t.loc[pl,'cascade_pd']:.3f}", ha="center", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"Pfa {pl}" for pl in res]); ax[0].set_ylim(0.8, 1.02); ax[0].set_ylabel("Pd (test images)"); ax[0].legend(loc="lower left"); ax[0].set_title("(a) Detection probability")
fa_w = []
for pl in res:
    tst = per_img[per_img.split == "test"]; fa_w.append(tst[f"offShipEvents_pooled_{pl}"].mean())
ax[1].bar(x - w / 2, fa_w, w, color="#9aa7b8", edgecolor="k", lw=.5, label="Weibull only (off-ship events)"); ax[1].bar(x + w / 2, [t.loc[pl, "fa_per_img"] for pl in res], w, color="#2e8b57", edgecolor="k", lw=.5, label="after CNN (false accepted events)")
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"Pfa {pl}" for pl in res]); ax[1].set_ylabel("false events per image (log)"); ax[1].legend(fontsize=8); ax[1].set_title("(b) False events per image")
for xi, v in zip(x - w / 2, fa_w): ax[1].text(xi, v * 1.15, f"{v:.0f}", ha="center", fontsize=8)
for xi, pl in zip(x + w / 2, res): ax[1].text(xi, t.loc[pl, "fa_per_img"] * 1.15, f"{t.loc[pl,'fa_per_img']:.2f}", ha="center", fontsize=8)
fig.suptitle(f"Test split (842 images): Weibull-only prescreen vs the full cascade (retention target {a.target:.0%})", fontsize=10); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_weibull_vs_cascade.png")); plt.close(fig)

# (3) operating curves after the CNN
fig, ax = plt.subplots(figsize=(7, 4.6))
for pl in res:
    s = cas[(cas.subset == "test") & (cas.pfa == pl)].sort_values("fa_per_img")
    ax.plot(s.fa_per_img, s.cascade_pd, "o-", color=COL[pl], label=f"Pfa {pl}")
ax.set_xscale("log"); ax.set_xlabel("false accepted events per image (log)"); ax.set_ylabel("cascade Pd (test)"); ax.legend(); ax.set_title("Cascade operating curves (retention targets 80-99 %)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cascade_operating_curves.png")); plt.close(fig)
print("saved cascade outputs to", OUT)
