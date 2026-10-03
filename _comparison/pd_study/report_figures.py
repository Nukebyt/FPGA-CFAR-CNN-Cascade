# -*- coding: utf-8 -*-
"""Figures + numbers for the Weibull-only (pooled prescreen) report.  Output: Results/pd_study/report/
Uses the config-A candidate set (HWDATA=pooldetA: pooled 2x2 log-mean, window 25/17, Pfa 0.03, gate 0.6, 5x5 peak events), all 5,604 images.
usage: python report_figures.py"""
import json
import os
import sys

os.environ["HWDATA"] = "pooldetA"
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "_comparison", "cnn"))
import hwlib as H                                                    # noqa: E402

RES = H.RES
OUT = os.path.join(RES, "pd_study", "report")
os.makedirs(OUT, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, GREY, RED, GREEN, ORANGE = "#1f4e9c", "#8c8c8c", "#c0392b", "#2e8b57", "#e08a1e"


def wilson(k, n):
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# ------------------------------------------------------------------ data: events + ships
d = H.HWData()
m = np.load(os.path.join(H.CACHE, "meta.npz")); gt2 = m["gtIdx2"].astype(np.int64)
S = pd.read_csv(os.path.join(RES, "pd_study", "phase1_ship_table.csv"), low_memory=False)
S["gt"] = S.groupby("img")["annIdx"].rank(method="first").astype(int)
S["key"] = (S["img"].astype(int) - 1) * 256 + S["gt"]
k1 = (d.img * 256 + d.gt)[d.labels & (d.gt > 0)]; k2 = (d.img * 256 + gt2)[gt2 > 0]
S["delivered"] = S["key"].isin(set(np.concatenate([k1, k2]).tolist()))
n_ev_img = np.bincount(d.img, minlength=len(d.nships))
res = dict(n_ships=int(len(S)), pd_all=float(S.delivered.mean()), missed=int((~S.delivered).sum()))

# ------------------------------------------------------------------ tolerance sensitivity (distance event -> ship polygon)
ann = json.load(open(os.path.join(ROOT, "HRSID", "annotations", "train_test2017.json")))
id2name = {i["id"]: i["file_name"] for i in ann["images"]}
by = {}
for ai, a in enumerate(ann["annotations"]):
    by.setdefault(id2name[a["image_id"]], []).append(ai)
names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
cy = np.array(m["cxy"])  # (2,N): full-resolution (y,x), 1-based
order = np.argsort(d.img, kind="stable"); starts = np.searchsorted(d.img[order], np.arange(len(names) + 1))
ship_dist = {}
for ii, nm in enumerate(names):
    ids = by.get(nm, [])
    if not ids:
        continue
    ev = order[starts[ii]:starts[ii + 1]]
    ey = np.clip(np.round(cy[0, ev]).astype(int) - 1, 0, 799) if len(ev) else np.zeros(0, int)
    ex = np.clip(np.round(cy[1, ev]).astype(int) - 1, 0, 799) if len(ev) else np.zeros(0, int)
    for r, ai in enumerate(ids, 1):
        a = ann["annotations"][ai]
        if not len(ev):
            ship_dist[(ii, r)] = np.inf; continue
        im = Image.new("L", (800, 800), 0); dr = ImageDraw.Draw(im)
        seg = a["segmentation"]; seg = seg[0] if isinstance(seg[0], list) else seg
        dr.polygon([(seg[i], seg[i + 1]) for i in range(0, len(seg) - 1, 2)], fill=1)
        mk = np.array(im, bool)
        if not mk.any():
            x0, y0, w0, h0 = [int(round(v)) for v in a["bbox"]]; mk[y0:y0 + h0, x0:x0 + w0] = True
        dist = ndimage.distance_transform_edt(~mk)
        ship_dist[(ii, r)] = float(dist[ey, ex].min())
S["dist"] = [ship_dist.get((int(i) - 1, int(g)), np.inf) for i, g in zip(S["img"], S["gt"])]
tols = [0, 1, 2, 3, 4, 6, 8, 12]
tol_pd = [float((S["dist"] <= t).mean()) for t in tols]
res["tolerance"] = dict(zip(map(str, tols), tol_pd))
print("Pd vs event-to-polygon tolerance (px):", {t: round(v, 4) for t, v in zip(tols, tol_pd)})

# ------------------------------------------------------------------ Fig: tolerance + per-image histogram + split/scene
im = S.groupby("img").agg(ships=("delivered", "size"), found=("delivered", "sum"), scene=("scene", "first"), split=("split", "first")).reset_index()
im["pd"] = im.found / im.ships
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
ax[0].plot(tols, tol_pd, "o-", color=BLUE); ax[0].axvline(4, color=GREY, ls=":"); ax[0].set_xlabel("max. distance event -> ship polygon (px, full resolution)")
ax[0].set_ylabel("ship Pd"); ax[0].set_ylim(0.9, 1.002); ax[0].set_title("(a) Sensitivity to the hit tolerance")
for t, v in zip(tols, tol_pd):
    if t in (0, 2, 4, 8):
        ax[0].annotate(f"{v:.4f}", (t, v), textcoords="offset points", xytext=(4, -12), fontsize=7)
ax[1].hist(im.pd, bins=np.linspace(0, 1.0001, 41), color=BLUE); ax[1].set_yscale("log"); ax[1].set_xlabel("per-image Pd"); ax[1].set_ylabel("images (log scale)")
ax[1].set_title(f"(b) {len(im)} images: {100*(im.pd==1).mean():.2f} % complete")
cats = [("all", S), ("train", S[S.split == "train"]), ("val", S[S.split == "val"]), ("test", S[S.split == "test"]), ("offshore", S[S.scene == "offshore"]), ("inshore", S[S.scene == "inshore"])]
for i, (lab, g) in enumerate(cats):
    k, n = int(g.delivered.sum()), len(g); lo, hi = wilson(k, n)
    ax[2].bar(i, k / n, color=BLUE if lab in ("all", "train", "val", "test") else (GREEN if lab == "offshore" else RED), yerr=[[max(k / n - lo, 0)], [max(hi - k / n, 0)]], capsize=3)
    ax[2].text(i, 0.985, f"{k/n:.4f}\n(n={n})", ha="center", fontsize=6.5, color="k")
ax[2].set_xticks(range(len(cats))); ax[2].set_xticklabels([c[0] for c in cats]); ax[2].set_ylim(0.98, 1.001); ax[2].set_ylabel("ship Pd"); ax[2].set_title("(c) By split and scene (95 % Wilson CI)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fullsweep_overview.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig: by factor
BINS = [("areaMask", "Ship area (polygon px)", [0, 25, 50, 100, 250, 500, 1e9]), ("dxMax", "Contrast dx = x_max - mu_bg (nat)", [-9, 0.8, 1.0, 1.2, 1.5, 1.8, 9]),
        ("nbrDist", "Gap to nearest ship (px)", [-1, 0.5, 5, 20, 60, 1e9]), ("distBorder", "Distance to image edge (px)", [-1, 8, 16, 32, 64, 1e9]),
        ("sdBg", "Local clutter s.d. (log domain)", [0, 0.3, 0.35, 0.4, 0.5, 9])]
fig, axs = plt.subplots(1, 5, figsize=(16, 3.6))
rows = []
for a, (col, title, edges) in zip(axs, BINS):
    b = pd.cut(S[col].replace(np.inf, 1e9), edges, right=False)
    g = S.groupby(b, observed=True)["delivered"].agg(["sum", "size"])
    pdv = g["sum"] / g["size"]; ci = np.array([wilson(k, n) for k, n in zip(g["sum"], g["size"])])
    a.bar(range(len(g)), pdv, color=BLUE, yerr=[np.maximum(pdv - ci[:, 0], 0), np.maximum(ci[:, 1] - pdv, 0)], capsize=2)
    labs = []
    for iv in g.index:
        lo, hi = iv.left, iv.right
        labs.append(f"<{hi:g}" if lo <= 0 and col != "dxMax" and col != "nbrDist" else (f">={lo:g}" if hi >= 1e9 else f"{lo:g}-{hi:g}"))
    a.set_xticks(range(len(g))); a.set_xticklabels(labs, rotation=35, ha="right", fontsize=7)
    for i, n in enumerate(g["size"]):
        a.text(i, 0.905, f"n={n}", ha="center", fontsize=6, color="w")
        rows.append(dict(factor=col, bin=labs[i], n=int(n), pd=float(pdv.iloc[i]), lo=float(ci[i, 0]), hi=float(ci[i, 1])))
    a.set_ylim(0.9, 1.003); a.set_title(title, fontsize=8); a.set_ylabel("Pd" if a is axs[0] else "")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fullsweep_by_factor.png")); plt.close(fig)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "fullsweep_pd_by_factor.csv"), index=False)

# ------------------------------------------------------------------ Fig: design ladder (test split, 842 images / 2,877 ships)
T = pd.read_csv(os.path.join(RES, "pd_variants", "variants_test2.csv"))
steps = [("F17_13v", 1e-3, .75, "nms", "RTL baseline"), ("F17_13b", 1e-3, .75, "nms", "+ padded\nborder"), ("F17_13b", 1e-2, .75, "nms", "+ Pfa 1e-2"),
         ("F17_13b", 1e-2, .75, "peak5", "+ contrast-peak\nevents"), ("P2l17_13b", 1e-2, .75, "peak5", "+ 2x2 log-mean\npooling"),
         ("P2l25_17b", 1e-2, .75, "peak5", "+ window 25/17"), ("P2l25_17b", 3e-2, .6, "peak5", "+ Pfa 3e-2,\ngate 0.6  (A)")]
L = []
for g, pf, gt, ev, lab in steps:
    r = T[(T.group == g) & (T.pfa == pf) & (T.gate == gt) & (T.event == ev)].iloc[0]
    L.append((lab, r.pd_ev, r.pd_mask, r.ev_img))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(L))
ax[0].bar(x, [l[1] for l in L], color=[GREY] + [BLUE] * (len(L) - 2) + [GREEN]); ax[0].set_xticks(x); ax[0].set_xticklabels([l[0] for l in L], fontsize=7)
for i, l in enumerate(L):
    ax[0].text(i, l[1] + 0.004, f"{l[1]:.3f}", ha="center", fontsize=7)
ax[0].set_ylim(0.8, 1.01); ax[0].set_ylabel("prescreen Pd (event on the ship)"); ax[0].set_title("(a) Cumulative design steps (842 test images, 2,877 ships)")
ax[1].plot(x, [l[3] for l in L], "o-", color=ORANGE); ax[1].set_xticks(x); ax[1].set_xticklabels([l[0] for l in L], fontsize=7); ax[1].set_ylabel("candidate events per image (CNN workload)")
ax[1].set_ylim(0, 450); ax[1].set_title("(b) Candidate events per image")
for i, l in enumerate(L):
    ax[1].text(i, l[3] + 12, f"{l[3]:.0f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_design_ladder.png")); plt.close(fig)
pd.DataFrame([dict(step=l[0].replace("\n", " "), pd_event=l[1], pd_pixel=l[2], events_per_img=l[3]) for l in L]).to_csv(os.path.join(OUT, "design_ladder_test.csv"), index=False)

# ------------------------------------------------------------------ Fig: workload distribution
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.hist(n_ev_img[n_ev_img > 0], bins=np.logspace(0, 3.5, 40), color=BLUE); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("candidate events per image"); ax.set_ylabel("images"); ax.set_title(f"Event load (mean {n_ev_img.mean():.0f}, median {np.median(n_ev_img):.0f}, max {n_ev_img.max()})", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_event_load.png")); plt.close(fig)
res["events_per_img"] = dict(mean=float(n_ev_img.mean()), median=float(np.median(n_ev_img)), p95=float(np.quantile(n_ev_img, .95)), max=int(n_ev_img.max()))

# ------------------------------------------------------------------ Fig: pipeline schematic
fig, ax = plt.subplots(2, 1, figsize=(12, 3.6)); ax = ax.ravel()
def chain(a, items, title, color):
    a.axis("off"); a.set_xlim(0, len(items) * 1.6); a.set_ylim(0, 1); a.set_title(title, loc="left", fontsize=9)
    for i, t in enumerate(items):
        a.add_patch(plt.Rectangle((i * 1.6 + 0.05, 0.15), 1.35, 0.7, fc=color, ec="k", lw=.8))
        a.text(i * 1.6 + 0.725, 0.5, t, ha="center", va="center", fontsize=7)
        if i:
            a.annotate("", (i * 1.6 + 0.05, 0.5), (i * 1.6 - 0.2, 0.5), arrowprops=dict(arrowstyle="->"))
chain(ax[0], ["8-bit pixel\n800x800", "x = ln sqrt(I+0.5)", "ring 17x17 \\ 13x13\nc1, c2 (valid\ninterior only)", "Weibull threshold\nT = c1 + delta(c2, Pfa=1e-3)", "NMS corner of\neach run", "gate\nx - c1 >= 0.75"], "Baseline prescreen (RTL)", "#d9d9d9")
chain(ax[1], ["8-bit pixel\n800x800", "2x2 mean of\nln sqrt(I+0.5)\n(400x400)", "ring 25x25 \\ 17x17\nc1, c2 (padded\nborder)", "Weibull threshold\nT = c1 + delta(c2, Pfa=3e-2)", "gate\nx - c1 >= 0.6", "5x5 local maximum\nof contrast\n(peak event)"], "Proposed prescreen (configuration A)", "#cfe3c8")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pipeline.png")); plt.close(fig)

json.dump(res, open(os.path.join(OUT, "report_numbers.json"), "w"), indent=1)
print(json.dumps(res, indent=1)[:900])
