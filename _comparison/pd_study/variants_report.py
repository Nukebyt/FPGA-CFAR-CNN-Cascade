# -*- coding: utf-8 -*-
"""Aggregate pd_variants_eval.m output (Results/pd_variants/<prefix>_*.mat) into the comparison tables / figures of Phase 2-4.

metrics per variant (pooled over images):  Pd_ev = ships with an event on the polygon / ships   (the cascade Pd),  Pd_mask,
  events/img (CNN workload), offship/img (false events), pixel Pfa.  Bootstrap CIs resample images.
usage: python variants_report.py [--prefix tune_t] [--tag tune]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "Results", "pd_variants"))
BUDGETS = [300, 500, 1000, 2000]            # CNN workload budgets (gated events per image) for the "best Pd within budget" table


def load(prefix):
    Rs, meta = [], None
    for f in sorted(glob.glob(os.path.join(OUT, prefix + "*.mat"))):
        m = sio.loadmat(f)
        Rs.append(m["R"].astype(np.float64))
        if meta is None:
            meta = dict(groups=[str(x).strip() for x in np.array(m["meta"]["groups"][0, 0]).ravel()] if False else None)
            mm = m["meta"][0, 0]
            groups = [str(g[0]) if hasattr(g, "__len__") and not isinstance(g, str) else str(g) for g in mm["groups"].ravel()]
            meta = dict(groups=groups, pfas=mm["pfas"].ravel(), gates=mm["gates"].ravel(),
                        events=[str(e[0]) if hasattr(e, "__len__") and not isinstance(e, str) else str(e) for e in mm["events"].ravel()])
    return np.concatenate(Rs, axis=0), meta


def table(R, meta):
    n = R.shape[0]
    rows = []
    for g, gn in enumerate(meta["groups"]):
        for ip, pf in enumerate(meta["pfas"]):
            for ig, gt in enumerate(meta["gates"]):
                for ie, ev in enumerate(meta["events"]):
                    a = R[:, g, ip, ig, ie, :]
                    ns = a[:, 0].sum()
                    rows.append(dict(group=gn, pfa=pf, gate=gt, event=ev, pd_ev=a[:, 2].sum() / ns, pd_mask=a[:, 1].sum() / ns,
                                     ev_img=a[:, 3].mean(), off_img=a[:, 4].mean(), pix_pfa=a[:, 5].sum() / a[:, 6].sum(), n_ships=int(ns)))
    df = pd.DataFrame(rows)
    def fam(s):
        base = {"F": "full-res"}.get(s[0], None)
        if base is None:
            base = "pool" + s[1] + (" log-mean" if s[2] == "l" else " intensity-mean")
        return base + ("" if s.endswith("b") else " (RTL border)")
    df["family"] = df["group"].map(fam)
    return df


def boot_ci(R, g, ip, ig, ie, B=400, seed=0):
    rng = np.random.RandomState(seed); a = R[:, g, ip, ig, ie, :]; n = len(a); v = []
    for _ in range(B):
        s = rng.randint(0, n, n); v.append(a[s, 2].sum() / a[s, 0].sum())
    return np.percentile(v, [2.5, 97.5])


def main():
    R, meta = load(ARGS.prefix)
    df = table(R, meta)
    df.to_csv(os.path.join(OUT, f"variants_{ARGS.tag}.csv"), index=False)
    n_img = R.shape[0]
    print(f"{n_img} images, {int(df['n_ships'].iloc[0])} ships, {len(meta['groups'])} groups x {len(meta['pfas'])} Pfa x {len(meta['gates'])} gates x {len(meta['events'])} events")
    base = df[(df.group == "F17_13v") & (df.pfa == 1e-3) & (df.gate == 0.75) & (df.event == "nms")].iloc[0]
    print(f"\nBASELINE (RTL window + border, 1e-3, gate 0.75, nms): Pd_ev {base.pd_ev:.4f}  events/img {base.ev_img:.0f}  off-ship {base.off_img:.0f}")
    pad = df[(df.group == "F17_13b") & (df.pfa == 1e-3) & (df.gate == 0.75) & (df.event == "nms")].iloc[0]
    print(f"+ padded border:                                           Pd_ev {pad.pd_ev:.4f}  events/img {pad.ev_img:.0f}")

    pd.set_option("display.width", 220)
    # ---- per group at the baseline operating point and at 1e-2
    for pf in (1e-3, 1e-2):
        sub = df[(df.pfa == pf) & (df.gate == 0.75) & (df.event == "nms")].sort_values("pd_ev", ascending=False)
        print(f"\n== groups at Pfa {pf:g}, gate 0.75, event nms")
        print(sub[["group", "family", "pd_ev", "pd_mask", "ev_img", "off_img", "pix_pfa"]].round(4).to_string(index=False))

    # ---- best Pd within workload budgets, per event type
    print("\n== best Pd_ev within a CNN workload budget (events per image), any group / Pfa / gate")
    for bud in BUDGETS:
        for ev in meta["events"]:
            s = df[(df.ev_img <= bud) & (df.event == ev)].sort_values("pd_ev", ascending=False).head(1)
            r = s.iloc[0]
            print(f"  budget {bud:5d}  event {ev:4s}: Pd_ev {r.pd_ev:.4f}   {r.group}  Pfa {r.pfa:g}  gate {r.gate:g}  events/img {r.ev_img:.0f}  off-ship {r.off_img:.0f}")

    # ---- marginal effects (mean over everything else) of each design choice
    print("\n== effect of each design choice on Pd_ev at the budget-300 / Pfa 1e-2 / gate 0.75 / nms point")
    ref = df[(df.pfa == 1e-2) & (df.gate == 0.75) & (df.event == "nms")]
    print(ref.groupby("family")["pd_ev"].agg(["mean", "max", "count"]).round(4).to_string())
    for ev in meta["events"]:
        s = df[(df.pfa == 1e-2) & (df.gate == 0.75) & (df.event == ev) & (df.group == "F17_13b")].iloc[0]
        print(f"  event {ev:4s} (F17_13b, 1e-2, gate .75): Pd_ev {s.pd_ev:.4f} events/img {s.ev_img:.0f}")
    for gt in meta["gates"]:
        s = df[(df.pfa == 1e-2) & (df.gate == gt) & (df.event == "nms") & (df.group == "F17_13b")].iloc[0]
        print(f"  gate {gt:g} (F17_13b, 1e-2, nms): Pd_ev {s.pd_ev:.4f} events/img {s.ev_img:.0f}")

    # ---- figure: Pareto cloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam_col = {"full-res": "#1f77b4", "full-res (RTL border)": "#7f7f7f", "pool2 intensity-mean": "#ff7f0e", "pool2 log-mean": "#d62728",
               "pool4 intensity-mean": "#9467bd", "pool4 log-mean": "#2ca02c"}
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    for fam, g in df.groupby("family"):
        ax[0].scatter(g["ev_img"], g["pd_ev"], s=9, alpha=.45, color=fam_col.get(fam, "k"), label=fam)
    ax[0].scatter([base.ev_img], [base.pd_ev], s=120, marker="*", color="k", zorder=5, label="RTL baseline")
    ax[0].set_xscale("log"); ax[0].set_xlabel("candidate events per image (CNN workload)"); ax[0].set_ylabel("prescreen Pd (event on the ship)")
    ax[0].axhline(0.95, color="g", ls=":"); ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=7); ax[0].set_title(f"All {len(df)} variants, {n_img} tuning images")
    # frontier
    fr = df.sort_values("ev_img")
    best = -1; px = []; py = []; pl = []
    for _, r in fr.iterrows():
        if r.pd_ev > best + 1e-9:
            best = r.pd_ev; px.append(r.ev_img); py.append(r.pd_ev); pl.append(f"{r.group}/{r.pfa:g}/{r.gate:g}/{r.event}")
    ax[1].step(px, py, where="post", color="k"); ax[1].scatter(px, py, s=12, color="k")
    for x_, y_, l_ in list(zip(px, py, pl))[::max(1, len(px) // 12)]:
        ax[1].annotate(l_, (x_, y_), fontsize=6, rotation=20, xytext=(3, -8), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("candidate events per image"); ax[1].set_ylabel("best Pd_ev"); ax[1].grid(alpha=.3, which="both")
    ax[1].axhline(0.95, color="g", ls=":"); ax[1].set_title("Pareto frontier (Pd vs CNN workload)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_variants_{ARGS.tag}_pareto.png"), dpi=150); plt.close(fig)
    print("\nsaved", os.path.join(OUT, f"variants_{ARGS.tag}.csv"), "and the Pareto figure")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="tune2_u")
    ap.add_argument("--tag", default="tune2")
    ARGS = ap.parse_args()
    main()
