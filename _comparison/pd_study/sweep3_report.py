# -*- coding: utf-8 -*-
"""Whole-data-set (5,604 images, 16,951 ships) Weibull-only results at Pfa 1e-2, 1e-3, 1e-4 for two prescreens
(pooled 2x2 log-mean 17/13 vs full-resolution 17/13, both padded border, gate 0.75, 5x5 contrast-peak events).
Inputs : Results/pd_sweep3/part*.mat (pd_sweep3.m)      Outputs: Results/pd_sweep3/report/  (tables .csv, figures .png, numbers .json)
usage: python sweep3_report.py"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "_comparison", "Results")
OUT = os.path.join(RES, "pd_sweep3", "report"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "_comparison", "cnn"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150, "savefig.bbox": "tight"})

PF = [1e-2, 1e-3, 1e-4]; PFL = ["1e-2", "1e-3", "1e-4"]
VAR = [("pooled", p) for p in range(3)] + [("full", p) for p in range(3)]           # variant order of pd_sweep3.m
COL = {"1e-2": "#1f4e9c", "1e-3": "#4a86d0", "1e-4": "#9cc0ea"}
FLOOR = 1e-8


def wilson(k, n):
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(c - h, 0), min(c + h, 1)


# ------------------------------------------------------------------ load
Ss, Is = [], []
for f in sorted(glob.glob(os.path.join(RES, "pd_sweep3", "part*.mat"))):
    m = sio.loadmat(f); Ss.append(m["S"]); Is.append(m["Im"])
S = np.concatenate(Ss); Im = np.concatenate(Is)
names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
ins = {a["file_name"] for a in json.load(open(os.path.join(ROOT, "HRSID", "inshore_offshore", "inshore.json")))["images"]}
import hwlib as H                                                         # noqa: E402  (split definition)
n_all = len(names); order = np.arange(n_all); np.random.RandomState(H.SPLIT_SEED).shuffle(order)
ntr, nva = int(0.7 * n_all), int(0.15 * n_all)
split = np.empty(n_all, object); split[order[:ntr]] = "train"; split[order[ntr:ntr + nva]] = "val"; split[order[ntr + nva:]] = "test"

ship = pd.DataFrame({"img": S[:, 0].astype(int), "rank": S[:, 1].astype(int)})
for v, (g, p) in enumerate(VAR):
    ship[f"dist_{g}_{PFL[p]}"] = S[:, 2 + 2 * v]; ship[f"mask_{g}_{PFL[p]}"] = S[:, 3 + 2 * v]
ship["name"] = [names[i - 1] for i in ship.img]; ship["scene"] = np.where(ship["name"].isin(ins), "inshore", "offshore"); ship["split"] = split[ship.img - 1]
att = pd.read_csv(os.path.join(RES, "pd_study", "phase1_ship_table.csv"), low_memory=False)
att["rank"] = att.groupby("img")["annIdx"].rank(method="first").astype(int)
ship = ship.merge(att[["img", "rank", "areaMask", "dxMax", "nbrDist", "distBorder", "sdBg"]], on=["img", "rank"], how="left")
im = pd.DataFrame({"img": Im[:, 0].astype(int), "ships": Im[:, 1].astype(int), "bgPix": Im[:, 2]})
for v, (g, p) in enumerate(VAR):
    k = 3 + 4 * v; tag = f"{g}_{PFL[p]}"
    im[f"events_{tag}"] = Im[:, k]; im[f"off_{tag}"] = Im[:, k + 1]; im[f"falsePix_{tag}"] = Im[:, k + 2]; im[f"detPix_{tag}"] = Im[:, k + 3]
im["name"] = [names[i - 1] for i in im.img]; im["scene"] = np.where(im["name"].isin(ins), "inshore", "offshore"); im["split"] = split[im.img - 1]
assert len(ship) == 16951 and len(im) == 5604, (len(ship), len(im))

# ------------------------------------------------------------------ per-image table
rows = im[["img", "name", "split", "scene", "ships"]].copy(); rows["bgPix"] = im.bgPix.values
for g, p in VAR:
    tag = f"{g}_{PFL[p]}"
    found = ship.groupby("img")[f"dist_{tag}"].apply(lambda s: (s <= 4).sum()).reindex(im.img).values
    rows[f"Pd_{tag}"] = found / im.ships.values
    rows[f"pixelPfa_{tag}"] = im[f"falsePix_{tag}"].values / im.bgPix.values
    rows[f"eventPfa_{tag}"] = im[f"off_{tag}"].values / im.bgPix.values
    rows[f"events_{tag}"] = im[f"events_{tag}"].values; rows[f"offShipEvents_{tag}"] = im[f"off_{tag}"].values
    rows[f"log10pixelPfa_{tag}"] = np.log10(np.maximum(rows[f"pixelPfa_{tag}"], FLOOR))
rows.to_csv(os.path.join(OUT, "per_image_weibull_only.csv"), index=False, float_format="%.6g")

# ------------------------------------------------------------------ summary table
summ = []
for g, p in VAR:
    tag = f"{g}_{PFL[p]}"; hit = (ship[f"dist_{tag}"] <= 4)
    r = dict(design="pooled 2x2 log-mean 17/13" if g == "pooled" else "full-res 17/13", pfa=PF[p], pd_event=hit.mean(), pd_pixel=ship[f"mask_{tag}"].mean(), missed=int((~hit).sum()),
             images_complete=float((rows[f"Pd_{tag}"] == 1).mean()), events_mean=im[f"events_{tag}"].mean(), events_median=im[f"events_{tag}"].median(),
             events_p95=im[f"events_{tag}"].quantile(.95), off_events_mean=im[f"off_{tag}"].mean(),
             pixel_pfa=im[f"falsePix_{tag}"].sum() / im.bgPix.sum(), event_pfa=im[f"off_{tag}"].sum() / im.bgPix.sum())
    for sp in ("train", "val", "test"):
        r[f"pd_{sp}"] = hit[ship.split == sp].mean()
    for sc in ("inshore", "offshore"):
        r[f"pd_{sc}"] = hit[ship.scene == sc].mean()
    lo, hi = wilson(int(hit.sum()), len(hit)); r["ci_lo"], r["ci_hi"] = lo, hi
    summ.append(r)
summ = pd.DataFrame(summ); summ.to_csv(os.path.join(OUT, "summary_weibull_only.csv"), index=False, float_format="%.5g")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(summ.round(4).to_string(index=False))
json.dump(summ.to_dict("records"), open(os.path.join(OUT, "summary_weibull_only.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ Fig 1: designs x Pfa on the whole data set
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3)); x = np.arange(2); w = .26
for j, pl in enumerate(PFL):
    pdv = [summ[(summ.design.str.startswith(d)) & (summ.pfa == PF[j])].pd_event.iloc[0] for d in ("full", "pooled")]
    ev = [summ[(summ.design.str.startswith(d)) & (summ.pfa == PF[j])].events_mean.iloc[0] for d in ("full", "pooled")]
    ax[0].bar(x + (j - 1) * w, pdv, w, color=COL[pl], edgecolor="k", lw=.5, label=f"Pfa = {pl}"); ax[1].bar(x + (j - 1) * w, ev, w, color=COL[pl], edgecolor="k", lw=.5, label=f"Pfa = {pl}")
    for xi, v in zip(x + (j - 1) * w, pdv): ax[0].text(xi, v + .004, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + (j - 1) * w, ev): ax[1].text(xi, v + 5, f"{v:.0f}", ha="center", fontsize=8)
for a in ax: a.set_xticks(x); a.set_xticklabels(["full resolution\n+ contrast-peak events", "2x2 log-mean pooling\n+ contrast-peak events"])
ax[0].set_ylim(0.5, 1.02); ax[0].set_ylabel("Weibull-only Pd (event within 4 px of the ship)"); ax[0].set_title("(a) Pd, all 16,951 ships"); ax[0].legend(loc="lower right")
ax[1].set_ylabel("mean candidate events per image"); ax[1].set_title("(b) Events the CNN has to check (5,604 images)")
fig.suptitle("Weibull-only prescreen over the whole data set, window 17/13, gate 0.75", fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_compare_designs.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 2: per-image Pd and Pfa (log), pooled design, 3 Pfa
def rolling(y, wdw):
    k = np.ones(wdw); return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")
fig, axs = plt.subplots(3, 3, figsize=(16, 11))
for r_, (pl, p) in enumerate(zip(PFL, range(3))):
    tag = f"pooled_{pl}"; order_ = np.argsort(rows[f"pixelPfa_{tag}"].values); xr = np.arange(len(rows))
    for c_, (key, lab, logy) in enumerate([(f"Pd_{tag}", "Pd per image", False), (f"pixelPfa_{tag}", "pixel Pfa per image (log)", True), (f"eventPfa_{tag}", "event Pfa per image (log)", True)]):
        y = rows[key].values[order_]; yy = np.maximum(y, FLOOR) if logy else y
        axs[r_, c_].plot(xr, yy, ".", ms=1.6, color=COL[pl] if logy or True else "k", alpha=.5)
        sm = rolling(y, 201); axs[r_, c_].plot(xr, np.maximum(sm, FLOOR) if logy else sm, "-", color="k", lw=1.8, label="rolling mean (201 images)")
        if logy: axs[r_, c_].set_yscale("log"); axs[r_, c_].set_ylim(FLOOR / 3, 0.2)
        else: axs[r_, c_].set_ylim(-0.03, 1.03)
        axs[r_, c_].set_ylabel(f"Pfa {pl}: {lab}"); axs[r_, c_].grid(alpha=.3, which="both")
        if r_ == 2: axs[r_, c_].set_xlabel("image rank (sorted by this setting's pixel Pfa)")
axs[0, 0].legend(fontsize=7, loc="lower left")
fig.suptitle("Per-image Weibull-only results, pooled 2x2 log-mean prescreen, 5,604 images (floor 1e-8 = no false pixels / events)", fontsize=10); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_per_image_pooled_3pfa.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 3: per-image false-alarm distribution (log), designs x Pfa
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
bins = np.logspace(-8, -1, 36)
for pl in PFL:
    for g, ls in (("pooled", "-"), ("full", "--")):
        v = rows[f"pixelPfa_{g}_{pl}"].values; ax[0].hist(np.maximum(v, FLOOR), bins=bins, histtype="step", lw=1.6, ls=ls, color=COL[pl], label=f"{g} Pfa {pl}")
        e = rows[f"events_{g}_{pl}"].values; ax[1].hist(e[e > 0], bins=np.logspace(0, 3.6, 36), histtype="step", lw=1.6, ls=ls, color=COL[pl], label=f"{g} Pfa {pl}")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("per-image pixel Pfa (log)"); ax[0].set_ylabel("images"); ax[0].set_title("(a) Pixel false-alarm rate per image")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("candidate events per image"); ax[1].set_ylabel("images"); ax[1].set_title("(b) Event load per image"); ax[0].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_per_image_distributions.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 4: overview per Pfa (tolerance, per-image Pd histogram, split/scene)
fig, axs = plt.subplots(3, 3, figsize=(14, 10)); tols = [0, 1, 2, 3, 4, 6, 8, 12]
for r_, pl in enumerate(PFL):
    tag = f"pooled_{pl}"; d_ = ship[f"dist_{tag}"].values
    tp = [float((d_ <= t).mean()) for t in tols]
    axs[r_, 0].plot(tols, tp, "o-", color=COL[pl]); axs[r_, 0].axvline(4, color="grey", ls=":"); axs[r_, 0].set_ylim(min(tp) - 0.02, 1.002)
    axs[r_, 0].set_ylabel(f"Pfa {pl}: ship Pd"); axs[r_, 0].set_title("(a) vs event-to-polygon tolerance (px)" if r_ == 0 else "")
    for t, v in zip(tols, tp):
        if t in (0, 4, 12): axs[r_, 0].annotate(f"{v:.4f}", (t, v), textcoords="offset points", xytext=(3, -11), fontsize=7)
    axs[r_, 1].hist(rows[f"Pd_{tag}"], bins=np.linspace(0, 1.0001, 41), color=COL[pl]); axs[r_, 1].set_yscale("log")
    axs[r_, 1].set_title(f"(b) per-image Pd: {100*(rows[f'Pd_{tag}']==1).mean():.1f} % complete" if r_ else f"(b) per-image Pd (log count): {100*(rows[f'Pd_{tag}']==1).mean():.1f} % complete", fontsize=8)
    cats = [("all", ship), ("train", ship[ship.split == "train"]), ("val", ship[ship.split == "val"]), ("test", ship[ship.split == "test"]), ("offshore", ship[ship.scene == "offshore"]), ("inshore", ship[ship.scene == "inshore"])]
    for i, (lab, g_) in enumerate(cats):
        k, n = int((g_[f"dist_{tag}"] <= 4).sum()), len(g_); lo, hi = wilson(k, n)
        axs[r_, 2].bar(i, k / n, color=COL[pl], edgecolor="k", lw=.5, yerr=[[max(k / n - lo, 0)], [max(hi - k / n, 0)]], capsize=2)
        axs[r_, 2].text(i, 0.0 + (k / n) * 0.0 + 0.02, f"{k/n:.3f}", ha="center", fontsize=7, rotation=90, color="w" if k / n > 0.3 else "k")
    axs[r_, 2].set_xticks(range(len(cats))); axs[r_, 2].set_xticklabels([c[0] for c in cats], fontsize=8); axs[r_, 2].set_ylim(0, 1.02)
    axs[r_, 2].set_title("(c) by split and scene (95 % CI)" if r_ == 0 else "")
axs[2, 0].set_xlabel("tolerance (px)"); axs[2, 1].set_xlabel("per-image Pd")
fig.suptitle("Pooled 2x2 log-mean prescreen, whole data set, three false-alarm settings", fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_overview_3pfa.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig 5: Pd by factor, 3 Pfa
BINS = [("areaMask", "Ship area (px)", [0, 25, 50, 100, 250, 500, 1e9]), ("dxMax", "Contrast dx (nat)", [-9, 0.8, 1.0, 1.2, 1.5, 1.8, 9]),
        ("nbrDist", "Gap to nearest ship (px)", [-1, 0.5, 5, 20, 60, 1e9]), ("distBorder", "Distance to image edge (px)", [-1, 8, 16, 32, 64, 1e9]), ("sdBg", "Local clutter s.d.", [0, 0.3, 0.35, 0.4, 0.5, 9])]
fig, axs = plt.subplots(1, 5, figsize=(18, 3.9)); fac_rows = []
for a, (col, title, edges) in zip(axs, BINS):
    b = pd.cut(ship[col].replace(np.inf, 1e9), edges, right=False); labs = None
    for j, pl in enumerate(PFL):
        g_ = ship.groupby(b, observed=True)[f"dist_pooled_{pl}"].agg(lambda s: (s <= 4).mean()); n_ = ship.groupby(b, observed=True).size()
        ci = np.array([wilson(int(round(v * n)), int(n)) for v, n in zip(g_.values, n_.values)])
        xs = np.arange(len(g_)) + (j - 1) * 0.26
        a.bar(xs, g_.values, 0.26, color=COL[pl], edgecolor="k", lw=.4, yerr=[np.maximum(g_.values - ci[:, 0], 0), np.maximum(ci[:, 1] - g_.values, 0)], capsize=1.5, label=f"Pfa {pl}")
        labs = [f"{iv.left:g}-{iv.right:g}" if iv.right < 1e9 else f">={iv.left:g}" for iv in g_.index]
        for i, (v, n) in enumerate(zip(g_.values, n_.values)): fac_rows.append(dict(factor=col, bin=labs[i], pfa=pl, n=int(n), pd=float(v)))
    a.set_xticks(range(len(labs))); a.set_xticklabels(labs, rotation=35, ha="right", fontsize=7); a.set_title(title, fontsize=8); a.set_ylim(0.7, 1.005)
axs[0].set_ylabel("Pd"); axs[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_by_factor_3pfa.png")); plt.close(fig)
pd.DataFrame(fac_rows).to_csv(os.path.join(OUT, "pd_by_factor_3pfa.csv"), index=False)
print("saved to", OUT)

# ------------------------------------------------------------------ per-image Pfa in decades ("log table")
edges = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0001]; lab_e = ["0 (none) .. <1e-6", "1e-6 .. 1e-5", "1e-5 .. 1e-4", "1e-4 .. 1e-3", "1e-3 .. 1e-2", "1e-2 .. 1e-1", ">= 1e-1"]
dec = []
for g, p in VAR:
    tag = f"{g}_{PFL[p]}"
    for metric in ("pixelPfa", "eventPfa"):
        v = rows[f"{metric}_{tag}"].values; cnt = np.histogram(v, bins=edges)[0]
        for l, c_ in zip(lab_e, cnt): dec.append(dict(design=g, pfa=PFL[p], metric=metric, per_image_range=l, images=int(c_), share=c_ / len(v)))
pd.DataFrame(dec).to_csv(os.path.join(OUT, "per_image_pfa_decade_table.csv"), index=False, float_format="%.5g")
print("decade table saved")
