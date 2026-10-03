# -*- coding: utf-8 -*-
"""Figures + measured numbers for the explainer report (contrast-peak events and 2x2 log-mean pooling).
A line-by-line numpy re-implementation of the MATLAB prescreen (pd_variants_eval2.m / extract_cnn_patches_pooldet.m) is used on one real HRSID image.
Output: Results/pd_study/explain/"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "_comparison", "Results", "pd_study", "explain"); os.makedirs(OUT, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})
X0, EULER = 1.2125, 0.5772156649015329


def boxsum(A, k):
    C = np.pad(np.cumsum(np.cumsum(A, 0), 1), ((1, 0), (1, 0)))
    return C[k:, k:] - C[:-k, k:] - C[k:, :-k] + C[:-k, :-k]


def ring_stats(x, sli, guard):
    TK, TG = (sli - 1) // 2, (guard - 1) // 2
    pad = lambda A: np.pad(A, TK, mode="symmetric")
    def ring(Ap):
        return boxsum(Ap, sli) - boxsum(Ap[TK - TG:Ap.shape[0] - (TK - TG), TK - TG:Ap.shape[1] - (TK - TG)], guard)
    y = x - X0
    N = ring(pad(np.ones_like(x))); S1 = ring(pad(y)); S2 = ring(pad(y * y))
    return X0 + S1 / N, np.maximum((S2 - S1 ** 2 / N) / (N - 1), 1e-9)


def blockmean(A, f):
    h, w = (A.shape[0] // f) * f, (A.shape[1] // f) * f
    return A[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def prescreen(I, pool, mode, sli, guard, pfa, gate, event):
    """I: 2-D intensity image.  Returns x, c1, detection map D, event mask E (domain pixels)."""
    xl = np.log(np.sqrt(I + 0.5))
    x = xl if pool == 1 else (blockmean(xl, pool) if mode == "log" else np.log(np.sqrt(blockmean(I, pool) + 0.5)))
    c1, c2 = ring_stats(x, sli, guard)
    C = np.clip(np.sqrt(np.pi ** 2 / 6 / c2), 0.8, 8.0)                       # Weibull shape from c2 (method of log-cumulants)
    delta = (EULER + np.log(-np.log(pfa))) / C
    D = x > c1 + delta
    contrast = x - c1
    Dg = D & (contrast >= gate)
    if event == "nms":
        up = np.zeros_like(D); up[1:] = D[:-1]; prv = np.zeros_like(D); prv[:, 1:] = D[:, :-1]
        upl = np.zeros_like(D); upl[:, 1:] = up[:, :-1]; upr = np.zeros_like(D); upr[:, :-1] = up[:, 1:]
        E = D & ~prv & ~up & ~upl & ~upr & Dg
    else:                                                                     # peak5: gated pixel whose contrast is the maximum of its 5x5 neighbourhood
        Cd = np.where(Dg, contrast, -np.inf)
        E = Dg & (Cd >= ndimage.maximum_filter(Cd, size=5, mode="constant", cval=-np.inf))
    return x, c1, c2, D, E, contrast


if __name__ == "__main__":
    ann = json.load(open(os.path.join(ROOT, "HRSID", "annotations", "train_test2017.json")))
    id2name = {i["id"]: i["file_name"] for i in ann["images"]}; by = {}
    for a in ann["annotations"]:
        by.setdefault(id2name[a["image_id"]], []).append(a)
    names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
    import pandas as pd
    S = pd.read_csv(os.path.join(ROOT, "_comparison", "Results", "pd_study", "phase1_ship_table.csv"), low_memory=False)
    cand = S[(S.reason.str.startswith("D masked: own")) & (S.areaMask > 250) & (S.areaMask < 1200) & (S.nShips <= 3) & (S.distBorder > 40)].copy()
    rng = np.random.RandomState(3); found = None
    for _, r in cand.sample(frac=1, random_state=3).iterrows():
        nm = r["name"]; I = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", nm)), float); I = I[..., 0] if I.ndim == 3 else I
        a = by[nm][int(r["annIdx"]) - 1 - min(int(a_["id"]) for a_ in by[nm]) if False else 0]
        # ship polygon of THIS annotation: match by bbox
        a = [q for q in by[nm] if abs(q["bbox"][0] - r["bx1"] + 1) < 1.5 and abs(q["bbox"][1] - r["by1"] + 1) < 1.5]
        if not a: continue
        seg = a[0]["segmentation"]; seg = seg[0] if isinstance(seg[0], list) else seg
        im = Image.new("L", (800, 800), 0); ImageDraw.Draw(im).polygon([(seg[i], seg[i + 1]) for i in range(0, len(seg) - 1, 2)], fill=1)
        mk = np.array(im, bool); dist = ndimage.distance_transform_edt(~mk)
        _, _, _, Db, Eb, _ = prescreen(I, 1, "int", 17, 13, 1e-3, 0.75, "nms")
        ey, ex = np.nonzero(Eb); hit_b = len(ey) and dist[ey, ex].min() <= 4
        xp, c1p, c2p, Dp, Ep, Cp = prescreen(I, 2, "log", 17, 13, 1e-3, 0.75, "peak5")
        py, px = np.nonzero(Ep); hit_p = len(py) and dist[np.clip((py * 2 + 1).astype(int), 0, 799), np.clip((px * 2 + 1).astype(int), 0, 799)].min() <= 4
        if (not hit_b) and hit_p:
            found = (nm, I, mk, Db, Eb, xp, Dp, Ep, c1p, r); break
    assert found, "no demo ship found"
    nm, I, mk, Db, Eb, xp, Dp, Ep, c1p, r = found
    ys, xs = np.nonzero(mk); cy, cx = int(ys.mean()), int(xs.mean()); half = 70
    y0, y1, x0, x1 = max(cy - half, 0), min(cy + half, 800), max(cx - half, 0), min(cx + half, 800)
    xf = np.log(np.sqrt(I + 0.5))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(xf[y0:y1, x0:x1], cmap="gray"); ax[0].contour(mk[y0:y1, x0:x1], [0.5], colors="cyan", linewidths=.8)
    ax[0].set_title("(a) Image (log amplitude), ship outline in cyan", fontsize=8)
    ax[1].imshow(xf[y0:y1, x0:x1], cmap="gray"); yy, xx = np.nonzero(Db[y0:y1, x0:x1]); ax[1].scatter(xx, yy, s=2, c="red", alpha=.6, marker="s")
    ey, ex = np.nonzero(Eb[y0:y1, x0:x1]); ax[1].scatter(ex, ey, s=60, facecolors="none", edgecolors="yellow", linewidths=1.5)
    ax[1].contour(mk[y0:y1, x0:x1], [0.5], colors="cyan", linewidths=.8); ax[1].set_title("(b) Old design: ship missed", fontsize=8)
    up_ = np.kron(xp, np.ones((2, 2)))[y0:y1, x0:x1]
    ax[2].imshow(up_, cmap="gray"); Dpu = np.kron(Dp.astype(int), np.ones((2, 2)))[y0:y1, x0:x1] > 0; yy, xx = np.nonzero(Dpu); ax[2].scatter(xx, yy, s=2, c="red", alpha=.6, marker="s")
    py, px = np.nonzero(Ep); sel = (py * 2 >= y0) & (py * 2 < y1) & (px * 2 >= x0) & (px * 2 < x1)
    ax[2].scatter(px[sel] * 2 + 1 - x0, py[sel] * 2 + 1 - y0, s=60, facecolors="none", edgecolors="lime", linewidths=1.5)
    ax[2].contour(mk[y0:y1, x0:x1], [0.5], colors="cyan", linewidths=.8); ax[2].set_title("(c) New design: ship found", fontsize=8)
    for a_ in ax: a_.axis("off")
    fig.suptitle(f"Real example: {nm}, ship area {int(r['areaMask'])} px; both designs Pfa 1e-3, gate 0.75", fontsize=9, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_example.png")); plt.close(fig)
    print("example:", nm, "area", r["areaMask"], "baseline events", int(Eb.sum()), "pooled events", int(Ep.sum()))

    # ---------------------------------------------------------- measured effect of pooling on speckle (variance / correlation)
    from scipy.ndimage import uniform_filter
    rs = np.random.RandomState(0); ratios = []; corr = []
    pool_names = [n for n in names if n in by and len(by[n]) <= 2]
    for nm2 in rs.choice(pool_names, 60, replace=False):
        J = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", nm2)), float); J = J[..., 0] if J.ndim == 3 else J
        x1_ = np.log(np.sqrt(J + 0.5)); x2_ = blockmean(x1_, 2)
        v1 = uniform_filter(x1_ ** 2, 17) - uniform_filter(x1_, 17) ** 2; v2 = uniform_filter(x2_ ** 2, 17) - uniform_filter(x2_, 17) ** 2
        m1, m2 = np.median(v1), np.median(v2); ratios.append(m1 / m2)
        a, b = x1_[:, :-1].ravel(), x1_[:, 1:].ravel(); corr.append(np.corrcoef(a, b)[0, 1])
    stats = dict(var_ratio_median=float(np.median(ratios)), var_ratio_p10=float(np.percentile(ratios, 10)), var_ratio_p90=float(np.percentile(ratios, 90)),
                 lag1_corr_median=float(np.median(corr)), n_images=60)
    print("pooling effect on local variance of x (full-res / pooled):", stats)
    json.dump(stats, open(os.path.join(OUT, "pooling_stats.json"), "w"), indent=1)

    # ---------------------------------------------------------- schematic figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    rng = np.random.RandomState(5); n = 120; xx = np.arange(n)
    ship = np.zeros(n); ship[50:70] = 1.1
    c_row = 0.35 * rng.randn(n) + ship                                      # contrast along one row: speckle sigma 0.35 + ship
    ax[0].plot(xx, c_row, color="k", lw=.8); ax[0].axhline(0.75, color="grey", ls="--", lw=.8); ax[0].axvspan(50, 70, color="cyan", alpha=.2)
    det = c_row > 0.75; runs = np.nonzero(det & ~np.r_[False, det[:-1]])[0]
    ax[0].scatter(runs, c_row[runs], s=70, facecolors="none", edgecolors="orange", linewidths=1.6, label="old: first pixel of each run")
    pk = [i for i in range(n) if det[i] and c_row[i] == c_row[max(0, i - 2):i + 3].max()]
    ax[0].scatter(pk, c_row[pk], s=70, marker="^", color="green", label="new: local maximum of contrast (+-2 px)")
    ax[0].set_title("(a) One row: contrast x - c1 (speckle s.d. 0.35, ship +1.1)", fontsize=8); ax[0].legend(fontsize=7); ax[0].set_xlabel("pixel"); ax[0].set_ylabel("contrast (nat)")
    xs_ = np.linspace(-3, 6, 400); g = lambda m, s: np.exp(-(xs_ - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
    ax[1].plot(xs_, g(0, 1), color="grey", label="sea, single look (s.d. 1)"); ax[1].plot(xs_, g(0, 0.5), color="blue", label="sea after 2x2 mean of logs (s.d. ~1/2)")
    ax[1].plot(xs_, g(3, 1), color="grey", ls="--", label="ship (same mean contrast)"); ax[1].plot(xs_, g(3, 0.5), color="blue", ls="--", label="ship after pooling")
    ax[1].set_title("(b) Averaging 4 samples halves the noise, the ship's mean stays: contrast/noise doubles", fontsize=8); ax[1].legend(fontsize=7); ax[1].set_xlabel("contrast in units of the single-look noise")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_schematic.png")); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    A = np.arange(16).reshape(4, 4) + 1
    ax[0].imshow(A, cmap="Blues"); [ax[0].text(j, i, f"a{A[i,j]}", ha="center", va="center", fontsize=8) for i in range(4) for j in range(4)]
    for k in (1.5,): ax[0].axhline(k, color="r"); ax[0].axvline(k, color="r")
    ax[0].set_title("4x4 pixels -> four 2x2 blocks", fontsize=8); ax[0].axis("off")
    ax[1].imshow(np.arange(4).reshape(2, 2) + 1, cmap="Blues"); [ax[1].text(j, i, f"mean of\nln(a) in\nblock {2*i+j+1}", ha="center", va="center", fontsize=8) for i in range(2) for j in range(2)]
    ax[1].set_title("pooled image: 2x2 pixels (each = 4 originals)", fontsize=8); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pooling.png")); plt.close(fig)
    print("figures written to", OUT)
