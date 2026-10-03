# -*- coding: utf-8 -*-
"""Phase 1 of PAPER2_PD_ROADMAP: why does the Weibull prescreen lose ships?   (v2: strict hit definitions + clean-sea attribution)

Hit definitions per ship (all measured, per plane):
    boxHit    any detected pixel inside the GT box                       -- lenient; what computePdPfa/metrics.py counts. At high Pfa a box
                                                                            "hits" on sea clutter alone (Pd -> 100 % at Pfa 0.1 is meaningless)
    maskHit   any detected pixel inside the ship polygon                 -- the ship itself was detected
    evMask    a GATED NMS trigger (x-c1 >= 0.75) on the ship polygon +-3 px -- the CNN actually receives an event on the ship  <- cascade Pd
    evComp    a gated trigger in any detection component touching the box  -- the label rule used to train the CNN
  float model planes Pfa 1e-1..1e-6 and the bit-exact hardware model (planes 1e-3..1e-6, hardware gate).

usage:  python analyze_misses.py [--parts "part_v2p*.mat"]       (edit CONFIG / classify() to change bins and taxonomy)
Outputs in Results/pd_study/: phase1_*.csv / json and fig_p1_*.png
"""
import argparse
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
OUT = os.path.join(RES, "pd_study")
HRSID = os.path.join(ROOT, "HRSID")
sys.path.insert(0, os.path.join(ROOT, "_comparison", "cnn"))

# ================================================================== CONFIG (edit me)
PFAS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6]          # float planes: columns hit1..8 / maskHit1..8 / evComp1..8 / evMask1..8
PFX = [1e-3, 1e-4, 1e-5, 1e-6]                                     # hardware planes: maskHitFx1..4 / evCompFx1..4 / evMaskFx1..4
P1 = PFAS.index(1e-3) + 1
MISS = "evMaskFx1"                                                 # the loss we study: hardware chain, plane 1e-3, event on the ship
BINS = {
    "ship size (mask px)": ("areaMask", [0, 25, 50, 100, 250, 500, 1e9]),
    "contrast dx = x_max - mu_bg (nat)": ("dxMax", [-9, 0.6, 0.8, 1.0, 1.2, 1.5, 9]),
    "local clutter sd (log domain)": ("sdBg", [0, 0.3, 0.35, 0.4, 0.5, 9]),
    "nearest neighbour gap (px)": ("nbrDist", [-1, 0.5, 5, 20, 60, 1e9]),
    "distance to image border (px)": ("distBorder", [-1, 4, 8, 16, 32, 64, 1e9]),
    "ships in the image": ("nShips", [0, 1, 2, 4, 8, 16, 1e9]),
}
# ======================================================================================


def load():
    S, Im = [], []
    for f in sorted(glob.glob(os.path.join(OUT, ARGS.parts))):
        m = sio.loadmat(f)
        Sn = [str(x[0]) if hasattr(x, "__len__") else str(x) for x in m["Sn"].ravel()]
        Imn = [str(x[0]) if hasattr(x, "__len__") else str(x) for x in m["Imn"].ravel()]
        S.append(pd.DataFrame(m["S"], columns=Sn)); Im.append(pd.DataFrame(m["Im"], columns=Imn))
    return pd.concat(S, ignore_index=True), pd.concat(Im, ignore_index=True)


def scene_and_split(S, Im):
    names = sorted(os.listdir(os.path.join(HRSID, "images")))            # == MATLAB dir() order (verified)
    ins = {a["file_name"] for a in json.load(open(os.path.join(HRSID, "inshore_offshore", "inshore.json")))["images"]}
    off = {a["file_name"] for a in json.load(open(os.path.join(HRSID, "inshore_offshore", "offshore.json")))["images"]}
    import hwlib as H
    n = len(names)
    order = np.arange(n); np.random.RandomState(H.SPLIT_SEED).shuffle(order)
    ntr, nva = int(0.7 * n), int(0.15 * n)
    split = np.empty(n, object); split[order[:ntr]] = "train"; split[order[ntr:ntr + nva]] = "val"; split[order[ntr + nva:]] = "test"
    for D in (S, Im):
        idx = D["img"].astype(int).values - 1
        D["name"] = [names[i] for i in idx]
        D["scene"] = ["inshore" if nm in ins else ("offshore" if nm in off else "?") for nm in D["name"]]
        D["split"] = split[idx]
    return S, Im


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def classify(df):
    """Mutually exclusive reason for every ship that does NOT deliver an on-ship event to the CNN (hardware model, Pfa 1e-3), in this order:
       A  border         no CFAR-valid pixel on the ship (outer 8 px are never evaluated)
       B  RTL border     the float model finds the ship (pixel on the polygon) but the bit-exact model does not. Checked: ALL such ships lie within 8 px of the image edge --
                         the float front end evaluates the border (padding), the RTL never evaluates the outer 8 px (CFAR valid region). NOT quantisation error.
       C  gate / NMS     the ship's pixels ARE detected but no gated trigger lands on the ship (gate x-c1 >= 0.75 or the trigger sits off the ship)
       D  masked         not detected, although against the CLEAN sea (c1, c2 from ship-free pixels) it would be:  marginClean > 0
                         split by what contaminates the reference ring at the ship's best pixel: own ship / neighbouring ship / bright clutter / other
       E  low contrast   not detected and below the clean-sea threshold:  near-miss (marginClean > -0.15 nat) or deep (invisible)"""
    miss = df[MISS] == 0
    r = pd.Series("", index=df.index, dtype=object)
    mc = df["marginClean"]
    own, oth, brt = df["ringOwn"].fillna(0), df["ringOther"].fillna(0), df["ringBright"].fillna(0)
    dom = pd.Series("other (variance / shape mismatch)", index=df.index, dtype=object)
    big = np.maximum.reduce([own, oth, brt]) >= 0.05
    dom[big & (own >= oth) & (own >= brt)] = "own ship in the ring (self-masking)"
    dom[big & (oth > own) & (oth >= brt)] = "neighbouring ship in the ring"
    dom[big & (brt > own) & (brt > oth)] = "bright clutter in the ring"
    r[:] = "E2 low contrast, deep (below clean-sea threshold by > 0.15 nat)"
    r[(mc <= 0) & (mc > -0.15)] = "E1 low contrast, near miss (clean-sea margin -0.15..0 nat)"
    mk = mc > 0
    r[mk] = "D masked: " + dom[mk]
    r[(df["maskHitFx1"] == 1)] = "C gate / NMS: detected but no gated trigger on the ship"
    r[(df[f"maskHit{P1}"] == 1) & (df["maskHitFx1"] == 0)] = "B RTL border: outer 8 px never evaluated (float model finds it)"
    r[~np.isfinite(df[f"margin{P1}"])] = "A border (no CFAR-valid pixel)"
    out = pd.Series("hit", index=df.index, dtype=object)
    out[miss] = r[miss]
    return out


def main():
    S, Im = load()
    S, Im = scene_and_split(S, Im)
    os.makedirs(OUT, exist_ok=True)
    P = lambda *a: print(*a, flush=True)
    res = {"n_images": int(len(Im)), "n_ships": int(len(S))}
    P(f"{len(Im)} images, {len(S)} ships")

    # ---------------------------------------------------------------- 1. Pd by definition vs plane
    rows = []
    for k, pf in enumerate(PFAS, 1):
        rows.append(dict(model="float", pfa=pf, boxHit=S[f"hit{k}"].mean(), maskHit=S[f"maskHit{k}"].mean(), evComp=S[f"evComp{k}"].mean(),
                         evMask=S[f"evMask{k}"].mean(), pix_pfa=Im[f"falsePix{k}"].sum() / Im[f"bgPix{k}"].sum(),
                         trig_img=Im[f"trig{k}"].mean(), gated_img=Im[f"gated{k}"].mean()))
    for q, pf in enumerate(PFX, 1):
        kk = PFAS.index(pf) + 1
        rows.append(dict(model="hardware", pfa=pf, boxHit=S[f"hitFx{q}"].mean(), maskHit=S[f"maskHitFx{q}"].mean(), evComp=S[f"evCompFx{q}"].mean(),
                         evMask=S[f"evMaskFx{q}"].mean(), pix_pfa=np.nan, trig_img=np.nan, gated_img=Im[f"gated{kk}"].mean()))
    curve = pd.DataFrame(rows)
    P("\n== Pd by hit definition (all ships)  [gated_img = candidate events/image the CNN must digest (float gate)]")
    P(curve.round(4).to_string(index=False))
    res["pd_by_definition"] = curve.to_dict("records"); curve.to_csv(os.path.join(OUT, "phase1_pd_by_definition.csv"), index=False)
    for sc in ("inshore", "offshore"):
        m = S["scene"] == sc
        P(f"  {sc}: hardware 1e-3  box {S.loc[m,'hitFx1'].mean():.4f}  mask {S.loc[m,'maskHitFx1'].mean():.4f}  evMask {S.loc[m,'evMaskFx1'].mean():.4f}")
    for sp in ("train", "val", "test"):
        m = S["split"] == sp
        P(f"  {sp}: hardware 1e-3  evMask {S.loc[m,'evMaskFx1'].mean():.4f}  (n={m.sum()})")

    # ---------------------------------------------------------------- 2. taxonomy
    S["reason"] = classify(S)
    miss = S[S["reason"] != "hit"]
    P(f"\n== ships that deliver NO on-ship event to the CNN (hardware, Pfa 1e-3): {len(miss)} of {len(S)}  ({100*len(miss)/len(S):.2f} %) -> prescreen Pd {1-len(miss)/len(S):.4f}")
    tax = miss["reason"].value_counts().sort_index()
    taxdf = pd.DataFrame({"ships": tax, "share_of_losses_%": 100 * tax / len(miss), "Pd_points_%": 100 * tax / len(S)})
    P(taxdf.round(2).to_string())
    for sc in ("inshore", "offshore"):
        m = miss[miss["scene"] == sc]; tot = (S["scene"] == sc).sum()
        P(f"  {sc}: {len(m)} of {tot} ships lost ({100*len(m)/tot:.1f} %)")
    res["taxonomy"] = taxdf.reset_index().rename(columns={"reason": "reason"}).to_dict("records"); taxdf.to_csv(os.path.join(OUT, "phase1_miss_taxonomy.csv"))
    # recoverability of the losses by relaxing the plane (hardware planes only exist at 1e-3..1e-6; float planes up to 1e-1)
    P("\n  of the lost ships, share that gets an on-ship event at a looser float Pfa:")
    for k, pf in enumerate(PFAS, 1):
        if pf >= 1e-3:
            P(f"    Pfa {pf:g}: {100*(miss[f'evMask{k}']==1).mean():.1f} %  (and {100*(miss[f'maskHit{k}']==1).mean():.1f} % detected on the polygon)")
    mgn = -miss["marginClean"].dropna()
    P(f"  clean-sea margin of the lost ships (nat; positive = would be detected with a clean background): median {miss['marginClean'].median():.3f}, "
      f"share > 0: {100*(miss['marginClean']>0).mean():.0f} %")

    # ---------------------------------------------------------------- 3. Pd by factor (miss = MISS)
    P(f"\n== prescreen Pd ({MISS}) by factor with 95 % Wilson CI")
    fr = []
    for title, (col, edges) in BINS.items():
        v = S[col].replace(np.inf, 1e9)
        b = pd.cut(v, edges, right=False)
        P(f"-- {title}")
        for iv, g in S.groupby(b, observed=True):
            k, n = int((g[MISS] == 1).sum()), len(g); lo, hi = wilson(k, n)
            P(f"   {str(iv):>18}: n={n:5d}  Pd {k/n:.3f} [{lo:.3f},{hi:.3f}]  lost {n-k:4d}")
            fr.append(dict(factor=title, bin=str(iv), n=n, pd=k / n, lo=lo, hi=hi))
    for sc in ("inshore", "offshore"):
        g = S[S["scene"] == sc]; k, n = int((g[MISS] == 1).sum()), len(g); lo, hi = wilson(k, n)
        P(f"-- scene {sc}: n={n} Pd {k/n:.3f} [{lo:.3f},{hi:.3f}]"); fr.append(dict(factor="scene", bin=sc, n=n, pd=k / n, lo=lo, hi=hi))
    pd.DataFrame(fr).to_csv(os.path.join(OUT, "phase1_pd_by_factor.csv"), index=False)

    # ---------------------------------------------------------------- 4. what predicts a loss
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import roc_auc_score
        feats = ["areaMask", "bw", "bh", "dxMax", "zMax", "sdBg", "muBg", "c2Bg", "nbrDist", "distBorder", "nShips", "ringOwn", "ringOther", "ringBright",
                 "mx", "sx", "pBright", "pDark", "meanC2"]
        X = S.merge(Im[["img", "mx", "sx", "pBright", "pDark", "meanC2"]], on="img", how="left")
        X["nbrDist"] = X["nbrDist"].replace(np.inf, 1e3)
        y = (X[MISS] == 0).astype(int).values
        Xf = X[feats].fillna(0).values
        tr = X["split"].isin(["train", "val"]).values; te = ~tr
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(Xf[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(Xf[te])[:, 1])
        pi = permutation_importance(clf, Xf[te], y[te], n_repeats=5, random_state=0, scoring="roc_auc")
        imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
        P(f"\n== what predicts a lost ship (gradient boosting, train+val -> test images): AUC {auc:.3f}")
        P("permutation importance (drop in AUC):"); P(imp.round(4).to_string())
        res["loss_predictor_auc"] = float(auc); res["loss_importance"] = imp.to_dict(); imp.to_csv(os.path.join(OUT, "phase1_loss_importance.csv"))
    except Exception as e:
        P("(factor attribution skipped:", repr(e), ")")

    # ---------------------------------------------------------------- 5. figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fl = curve[curve["model"] == "float"]; hw = curve[curve["model"] == "hardware"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.log10(fl["pfa"])
    for col, lab, c in (("boxHit", "box hit (lenient)", "#bbbbbb"), ("maskHit", "pixel on the ship polygon", "#4292c6"), ("evMask", "gated event on the ship (cascade Pd)", "#08306b")):
        ax[0].plot(x, fl[col], "o-", color=c, label="float: " + lab)
    ax[0].plot(np.log10(hw["pfa"]), hw["evMask"], "s--", color="#d62728", label="hardware-exact: gated event on the ship")
    ax[0].axhline(0.95, color="g", ls=":"); ax[0].set_xlabel("Weibull Pfa plane (log10)"); ax[0].set_ylabel("Pd"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    ax[0].set_title(f"Prescreen Pd by hit definition, {len(S)} ships / {len(Im)} images")
    ax[1].plot(fl["gated_img"], fl["evMask"], "o-", color="#08306b")
    for gx, py, pf in zip(fl["gated_img"], fl["evMask"], fl["pfa"]):
        ax[1].annotate(f"{pf:g}", (gx, py), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("gated candidate events / image (CNN workload)"); ax[1].set_ylabel("Pd (event on the ship)"); ax[1].grid(alpha=.3, which="both")
    ax[1].set_title("Cascade Pd bought per unit of CNN workload (float model)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_p1_pd_vs_pfa.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    lab = taxdf.index.tolist()
    ax.barh(range(len(lab)), taxdf["Pd_points_%"], color="#4292c6"); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
    for i, (v, n) in enumerate(zip(taxdf["Pd_points_%"], taxdf["ships"])):
        ax.text(v + 0.02, i, f"{v:.2f} Pd points  (n={n})", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Pd points lost (% of all ships)")
    ax.set_title(f"Where the prescreen loses ships (hardware model, Pfa 1e-3): Pd {1-len(miss)/len(S):.3f}, {len(miss)} of {len(S)} lost")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_p1_miss_taxonomy.png"), dpi=150); plt.close(fig)

    fr = pd.DataFrame(fr); facs = list(BINS)
    fig, axs = plt.subplots(2, 3, figsize=(16, 7.5))
    for a, f in zip(axs.ravel(), facs):
        g = fr[fr["factor"] == f]
        a.bar(range(len(g)), g["pd"], color="#4292c6", yerr=[g["pd"] - g["lo"], g["hi"] - g["pd"]], capsize=2)
        a.set_xticks(range(len(g))); a.set_xticklabels(g["bin"], rotation=30, ha="right", fontsize=7)
        for i, n in enumerate(g["n"]):
            a.text(i, 0.02, f"n={n}", ha="center", fontsize=6.5, color="w")
        a.set_ylim(0, 1.02); a.axhline(0.95, color="g", ls=":"); a.set_title(f, fontsize=9); a.grid(alpha=.3, axis="y")
    fig.suptitle(f"Prescreen Pd ({MISS}) by factor, 95 % Wilson CI", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_p1_pd_by_factor.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    hit = S[MISS] == 1
    ax.scatter(S.loc[hit, "deltaClean"], S.loc[hit, "dxMax"], s=3, alpha=.2, color="#4292c6", label="event on ship")
    ax.scatter(S.loc[~hit, "deltaClean"], S.loc[~hit, "dxMax"], s=6, alpha=.6, color="#d62728", label="lost")
    lim = [S["deltaClean"].min(), S["deltaClean"].max()]
    ax.plot(lim, lim, "k--", lw=1, label="dx = clean-sea threshold offset")
    ax.set_xlabel("clean-sea Weibull offset delta (nat)"); ax.set_ylabel("ship contrast dx = x_max - mu_bg (nat)")
    ax.legend(); ax.grid(alpha=.3); ax.set_title("Ship contrast vs the offset it must exceed (clean sea)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_p1_contrast_vs_delta.png"), dpi=150); plt.close(fig)

    S.to_csv(os.path.join(OUT, "phase1_ship_table.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "phase1_summary.json"), "w"), indent=1, default=float)
    P("\nsaved tables/figures to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="part_v2p*.mat")
    ARGS = ap.parse_args()
    main()
