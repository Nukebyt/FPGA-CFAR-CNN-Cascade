# -*- coding: utf-8 -*-
"""Per-image Pd / Pfa log + graphs for the 250-image sweep (board results: counts_hw.npz; add --sw for the software-model run).

Per image and Pfa plane, for Weibull-only and for the cascade at the three CNN operating points:
    Pd          ships detected / GT ships in that image (NaN if the image has no ship)
    Pfa         false detection pixels / background pixels of that image
    FA events   false-alarm events (triggers / accepted candidates whose component touches no GT box)
Outputs (Results/sweep250/):  per_image_hw.csv  fig_per_image_hw_plane<k>.png  fig_per_image_hw_all_planes.png

usage:  python per_image.py [--sw] [--sort index|pfa]        edit the CONFIG block for colours / plane / floor
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "Results", os.environ.get("SWEEP", "sweep250")))

# ============================================================ CONFIG (edit me)
PFA_FLOOR = 1e-8                    # images with exactly zero false pixels are drawn here (hollow marker)
MAIN_PLANE = 0                      # plane for the detailed figure (0 = 1e-3, the CNN's training point)
COLORS = {"Weibull": "#7f7f7f", "CNN 80%": "#9ecae1", "CNN 85%": "#4292c6", "CNN 90%": "#08306b"}
# =============================================================================


def load(tag):
    z = np.load(os.path.join(OUT, f"counts{tag}.npz"), allow_pickle=True)
    wc, cc, thetas = z["wc"], z["cc"], z["thetas"]
    ops = dict(zip(z["ops_names"].tolist(), z["ops_thr"].tolist()))
    names = [str(n) for n in z["names"]]
    tcol = {k: int(np.argmin(np.abs(thetas - v))) for k, v in ops.items()}
    return wc, cc, names, tcol, [float(p) for p in z["planes"]]


def per_image(wc, cc, tcol):
    """-> dict[(plane, system)] = dict(pd, pfa, fa) arrays over images"""
    n, nP = wc.shape[:2]
    res = {}
    for p in range(nP):
        tot, bg = wc[:, p, 1], wc[:, p, 3]
        with np.errstate(invalid="ignore", divide="ignore"):
            res[(p, "Weibull")] = dict(pd=wc[:, p, 0] / tot, pfa=wc[:, p, 2] / bg, fa=wc[:, p, 4])
            for s, t in tcol.items():
                res[(p, s)] = dict(pd=cc[:, p, t, 0] / tot, pfa=cc[:, p, t, 1] / bg, fa=cc[:, p, t, 2])
    return res


def write_csv(path, names, wc, res, planes, systems):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["idx", "image", "gt_ships"]
        for p, pl in enumerate(planes):
            for s in systems:
                tag = f"{s.replace(' ', '').replace('%', '')}@{pl:.0e}"
                hdr += [f"Pd[{tag}]", f"Pfa[{tag}]", f"FAevents[{tag}]"]
        w.writerow(hdr)
        for i, nm in enumerate(names):
            row = [i, nm, int(wc[i, 0, 1])]
            for p in range(len(planes)):
                for s in systems:
                    r = res[(p, s)]
                    row += ["" if np.isnan(r["pd"][i]) else f"{r['pd'][i]:.4f}", f"{r['pfa'][i]:.3e}", int(r["fa"][i])]
            w.writerow(row)


def plane_figure(plt, ax_pd, ax_pfa, res, p, systems, order, n):
    x = np.arange(n)
    for s in systems:
        r = res[(p, s)]
        c = COLORS[s]
        pd = r["pd"][order]
        # Pd of one image is a fraction k/ships: draw as markers (jitter-free), cascade smaller on top of Weibull
        ax_pd.scatter(x, pd, s=14 if s == "Weibull" else 9, color=c, alpha=0.9 if s != "Weibull" else 0.6,
                      label=s if s != "Weibull" else "Weibull only", linewidths=0)
        pfa = r["pfa"][order]
        zero = pfa <= 0
        ax_pfa.scatter(x[~zero], pfa[~zero], s=14 if s == "Weibull" else 9, color=c, alpha=0.9 if s != "Weibull" else 0.6, linewidths=0)
        ax_pfa.scatter(x[zero], np.full(zero.sum(), PFA_FLOOR), s=14, facecolors="none", edgecolors=c, linewidths=0.7)
    ax_pd.set_ylim(-0.05, 1.08); ax_pd.set_ylabel("Pd per image"); ax_pd.grid(alpha=0.3)
    ax_pfa.set_yscale("log"); ax_pfa.set_ylabel("Pfa per image (log)"); ax_pfa.grid(alpha=0.3, which="both")
    ax_pfa.set_ylim(PFA_FLOOR / 3, 3e-2)


def lines_figure(plt, res, planes, systems, n, tag, plane_ids):
    """selected planes only; one row per plane (Pd | Pfa log | false-alarm events log), dots joined by lines"""
    x = np.arange(n)
    fig, axs = plt.subplots(len(plane_ids), 3, figsize=(18, 4.2 * len(plane_ids)), sharex=True, squeeze=False)
    for r, p in enumerate(plane_ids):
        pl = f"{planes[p]:.0e}".replace("e-0", "e-")
        for s in systems:
            c = COLORS[s]; lab = "Weibull only" if s == "Weibull" else s
            kw = dict(color=c, marker="o", ms=3, lw=0.8, alpha=0.9)
            axs[r, 0].plot(x, res[(p, s)]["pd"], label=lab, **kw)
            axs[r, 1].plot(x, np.maximum(res[(p, s)]["pfa"], PFA_FLOOR), **kw)
            axs[r, 2].plot(x, np.maximum(res[(p, s)]["fa"].astype(float), 0.5), **kw)
        axs[r, 0].set_ylim(-0.05, 1.08); axs[r, 0].set_ylabel(f"plane {pl}: Pd per image")
        axs[r, 1].set_yscale("log"); axs[r, 1].set_ylabel("Pfa per image (log)"); axs[r, 1].set_ylim(PFA_FLOOR / 3, 3e-2)
        axs[r, 2].set_yscale("log"); axs[r, 2].set_ylabel("false-alarm events per image (log)")
        for c_ in range(3):
            axs[r, c_].grid(alpha=0.3, which="both")
    for c_ in range(3):
        axs[-1, c_].set_xlabel("image (test-set order)")
    axs[0, 0].legend(ncol=4, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.08), frameon=False)
    fig.suptitle(f"Per-image Pd / Pfa / false-alarm events, {n} HRSID test images (CNN logits measured on the FPGA); "
                 "Pfa = 1e-8 and events = 0.5 mean zero", fontsize=10, y=0.995)
    fig.tight_layout()
    path = os.path.join(OUT, f"fig_per_image{tag}_lines.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


def rolling(y, w):
    """centred rolling mean of the LINEAR values (so a Pfa average is a real pooled-style rate), shrinking window at the ends"""
    k = np.ones(w)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")


def clean_figure(plt, res, planes, n, tag, plane_ids, op, win):
    """Weibull-only vs ONE cascade operating point; images sorted by that plane's Weibull-only Pfa so lines read as trends;
    thin line + dots = per image, thick line = rolling mean over `win` neighbouring images"""
    x = np.arange(n)
    fig, axs = plt.subplots(len(plane_ids), 3, figsize=(18, 4.4 * len(plane_ids)), sharex=True, squeeze=False)
    for r, p in enumerate(plane_ids):
        pl = f"{planes[p]:.0e}".replace("e-0", "e-")
        order = np.argsort(res[(p, "Weibull")]["pfa"])
        for s, lab in (("Weibull", "Weibull only"), (op, f"Cascade, {op}")):
            c = COLORS[s]
            for k, (key, floor) in enumerate((("pd", None), ("pfa", PFA_FLOOR), ("fa", 0.5))):
                y = res[(p, s)][key][order].astype(float)
                ax = axs[r, k]
                yy = y if floor is None else np.maximum(y, floor)
                ax.plot(x, yy, "-o", color=c, ms=2.5, lw=0.5, alpha=0.45)
                sm = rolling(y, win)
                ax.plot(x, sm if floor is None else np.maximum(sm, floor), "-", color=c, lw=2.6, label=f"{lab} (rolling mean, {win} images)" if k == 0 else None)
        axs[r, 0].set_ylim(-0.05, 1.08); axs[r, 0].set_ylabel(f"plane {pl}: Pd per image")
        axs[r, 1].set_yscale("log"); axs[r, 1].set_ylabel("Pfa per image (log)"); axs[r, 1].set_ylim(PFA_FLOOR / 3, 3e-2)
        axs[r, 2].set_yscale("log"); axs[r, 2].set_ylabel("false-alarm events per image (log)")
        for c_ in range(3):
            axs[r, c_].grid(alpha=0.3, which="both")
    for c_ in range(3):
        axs[-1, c_].set_xlabel("image rank (sorted by Weibull-only Pfa of that plane)")
    axs[0, 0].legend(ncol=2, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.06), frameon=False)
    fig.suptitle(f"Per-image Pd / Pfa / false-alarm events, {n} HRSID test images (CNN logits measured on the FPGA); "
                 "thin = per image, thick = rolling mean; Pfa 1e-8 / events 0.5 = zero", fontsize=10, y=0.995)
    fig.tight_layout()
    path = os.path.join(OUT, f"fig_per_image{tag}_clean.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sw", action="store_true", help="software-model run instead of the board run")
    ap.add_argument("--sort", default="index", choices=["index", "pfa"])
    ap.add_argument("--lines", action="store_true", help="only the --planes planes, dots connected by lines (fig_per_image_hw_lines.png)")
    ap.add_argument("--clean", action="store_true", help="Weibull + ONE CNN point, images sorted by Weibull-only Pfa, with smoothed trend (fig_per_image_hw_clean.png)")
    ap.add_argument("--op", default="CNN 90%", help="operating point for --clean")
    ap.add_argument("--win", type=int, default=21, help="rolling-mean window (images) for --clean")
    ap.add_argument("--planes", type=int, nargs="+", default=[0, 1], help="plane indices for --lines (0=1e-3, 1=1e-4, ...)")
    a = ap.parse_args()
    tag = "" if a.sw else "_hw"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wc, cc, names, tcol, planes = load(tag)
    n = len(names)
    systems = ["Weibull"] + list(tcol)
    res = per_image(wc, cc, tcol)
    write_csv(os.path.join(OUT, f"per_image{tag}.csv"), names, wc, res, planes, systems)

    if a.clean:
        clean_figure(plt, res, planes, n, tag, a.planes, a.op, a.win)
        return
    if a.lines:
        lines_figure(plt, res, planes, systems, n, tag, a.planes)
        return
    src = "CNN logits measured on the FPGA" if not a.sw else "software model"
    label = lambda p: f"Pfa plane {planes[p]:.0e}".replace("e-0", "e-")
    # ---- detailed figure for one plane
    p = MAIN_PLANE
    order = np.arange(n) if a.sort == "index" else np.argsort(res[(p, "Weibull")]["pfa"])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2, 1]))
    plane_figure(plt, a1, a2, res, p, systems, order, n)
    a1.legend(ncol=4, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.07), markerscale=1.8, frameon=False)
    fig.suptitle(f"Per-image Pd / Pfa / false-alarm events, {n} HRSID test images, {label(p)} ({src})", fontsize=10, y=0.995)
    x = np.arange(n)
    for s in systems:
        fa = res[(p, s)]["fa"][order].astype(float)
        a3.scatter(x, np.maximum(fa, 0.5), s=14 if s == "Weibull" else 9, color=COLORS[s], alpha=0.9 if s != "Weibull" else 0.6, linewidths=0)
    a3.set_yscale("log"); a3.set_ylabel("false-alarm events per image (log)\n(0 drawn at 0.5)"); a3.grid(alpha=0.3, which="both")
    a3.set_xlabel("image (test-set order)" if a.sort == "index" else "image (sorted by Weibull-only Pfa)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_per_image{tag}_plane{p}.png"), dpi=170); plt.close(fig)

    # ---- all four planes
    fig, axs = plt.subplots(4, 2, figsize=(15, 11), sharex=True)
    for p in range(len(planes)):
        order = np.arange(n)
        plane_figure(plt, axs[p, 0], axs[p, 1], res, p, systems, order, n)
        axs[p, 0].set_title(label(p), fontsize=9, loc="left")
    axs[0, 0].legend(ncol=4, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.18), markerscale=1.6, frameon=False)
    axs[-1, 0].set_xlabel("image"); axs[-1, 1].set_xlabel("image")
    fig.suptitle(f"Per-image Pd (left) and Pfa, log (right): Weibull-only vs cascade, {n} images ({src}); hollow = zero false pixels", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_per_image{tag}_all_planes.png"), dpi=150); plt.close(fig)

    # ---- console log summary
    print(f"wrote per_image{tag}.csv ({n} images) and figures to {OUT}")
    print(f"images with no GT ship: {int((wc[:,0,1]==0).sum())}")
    for p in range(len(planes)):
        for s in systems:
            r = res[(p, s)]
            print(f"{label(p):>18} {s:>8}: median Pd {np.nanmedian(r['pd']):.2f}  images with Pd=1: {int(np.nansum(r['pd']==1)):3d}  "
                  f"median Pfa {np.median(r['pfa']):.2e}  images with 0 false px: {int((r['pfa']==0).sum()):3d}")


if __name__ == "__main__":
    main()
