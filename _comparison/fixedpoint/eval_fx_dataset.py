# -*- coding: utf-8 -*-
"""Whole-data-set (5,604 images, 16,951 ships) evaluation of the integer prescreen against the float prescreen.
Three arithmetic levels per configuration:
   float_exact : MATLAB-equivalent float model (mean of the exact logs, real sqrt/ln)           [= pd_sweep3.m pooled group]
   ideal_q8    : exact arithmetic but on the 8-bit pooled frame the hardware actually stores
   fixed       : the integer model prescreen_fx.prescreen (what the RTL computes)
Strict event Pd = a gated event within 4 full-resolution px of the ship polygon (polygon & bbox mask).
Usage: python eval_fx_dataset.py [--n N] [--workers 3] [--out Results/fixedpoint/eval.npz]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prescreen_fx as fx

ROOT = fx.ROOT
X0 = 1.2125
TOL = 4
# (label, sli, guard, pfa_sel, tau)
CONFIGS = [("A_25_17_3e-2", 25, 17, 0, 0.60),
           ("B_17_13_1e-2", 17, 13, 1, 0.75),
           ("B_17_13_1e-3", 17, 13, 2, 0.75),
           ("B_17_13_1e-4", 17, 13, 3, 0.75)]


def float_exact(I, cfg):
    h0, w0 = (I.shape[0] // 2) * 2, (I.shape[1] // 2) * 2
    xl = np.log(np.sqrt(I[:h0, :w0] + 0.5))
    x = xl.reshape(h0 // 2, 2, w0 // 2, 2).mean(axis=(1, 3))
    TK, TG = cfg.TK, cfg.TG
    pad = lambda A: np.pad(A, TK, mode="symmetric")
    def ring(Ap):
        return fx.box(Ap, cfg.sli) - fx.box(Ap[TK - TG:Ap.shape[0] - (TK - TG), TK - TG:Ap.shape[1] - (TK - TG)], cfg.guard)
    y = x - X0
    N = ring(pad(np.ones_like(x))); S1 = ring(pad(y)); S2 = ring(pad(y * y))
    c1 = X0 + S1 / N; c2 = np.maximum((S2 - S1 ** 2 / N) / (N - 1), 1e-9)
    C = np.clip(np.sqrt(np.pi ** 2 / 6 / c2), fx.CMIN, fx.CMAX)
    K = fx.EULER + math.log(-math.log(fx.PFAS[cfg.pfa_sel]))
    D = x > c1 + K / C
    con = x - c1
    Dg = D & (con >= cfg.tau)
    Cd = np.where(Dg, con, -np.inf)
    h, w = Cd.shape; r = cfg.event // 2
    Pd = np.full((h + 2 * r, w + 2 * r), -np.inf); Pd[r:r + h, r:r + w] = Cd
    mx = Pd[0:h, 0:w].copy()
    for dy in range(cfg.event):
        for dx in range(cfg.event):
            mx = np.maximum(mx, Pd[dy:dy + h, dx:dx + w])
    return dict(D=D, Dg=Dg, E=Dg & (Cd >= mx))


def load_ann():
    ann = json.load(open(os.path.join(ROOT, "HRSID", "annotations", "train_test2017.json")))
    id2name = {i["id"]: i["file_name"] for i in ann["images"]}
    by = {}
    for a in ann["annotations"]:
        by.setdefault(id2name[a["image_id"]], []).append(a)
    return by


def ship_masks(anns, h0, w0):
    ms = []
    for a in anns:
        b = a["bbox"]
        x1 = max(0, int(round(b[0]))); y1 = max(0, int(round(b[1]))); x2 = min(w0 - 1, int(round(b[0] + b[2]))); y2 = min(h0 - 1, int(round(b[1] + b[3])))
        bm = np.zeros((h0, w0), bool); bm[y1:y2 + 1, x1:x2 + 1] = True
        m = bm
        try:
            sg = a["segmentation"]; sg = sg[0] if isinstance(sg[0], list) else sg
            im = Image.new("L", (w0, h0), 0)
            ImageDraw.Draw(im).polygon([(sg[i], sg[i + 1]) for i in range(0, len(sg) - 1, 2)], fill=1)
            pm = np.array(im, bool) & bm
            if pm.sum() >= 3:
                m = pm
        except Exception:
            pass
        ms.append(m)
    return ms


def work(args):
    name, anns = args
    I = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", name)))
    I = I[..., 0] if I.ndim == 3 else I
    h0, w0 = (I.shape[0] // 2) * 2, (I.shape[1] // 2) * 2
    masks = ship_masks(anns, h0, w0)
    dists = [ndimage.distance_transform_edt(~m) for m in masks]
    Du = np.minimum.reduce(dists) if dists else np.full((h0, w0), np.inf)
    P = fx.pool_q8(I)
    out = {"nShips": len(masks), "cfg": []}
    for (lab, sli, guard, ps, tau) in CONFIGS:
        cfg = fx.Cfg(sli, guard, ps, tau)
        res = {"float_exact": float_exact(I.astype(np.float64), cfg), "ideal_q8": fx.prescreen_float_on_q8(P, cfg), "fixed": fx.prescreen(P, cfg)}
        row = {}
        for mname, r in res.items():
            ey, ex = np.nonzero(r["E"])
            ry, rx = 2 * ey + 1, 2 * ex + 1
            md = [float(d[ry, rx].min()) if len(ey) else np.inf for d in dists]
            nOff = int((Du[ry, rx] > TOL).sum()) if len(ey) else 0
            row[mname] = dict(md=md, nEv=int(len(ey)), nOff=nOff, nDet=int(r["D"].sum()), nGate=int(r["Dg"].sum()))
        f, i, q = res["float_exact"], res["ideal_q8"], res["fixed"]
        row["mis_fixed_vs_ideal"] = int((q["D"] != i["D"]).sum()); row["mis_ideal_vs_float"] = int((i["D"] != f["D"]).sum())
        row["mis_fixed_vs_float"] = int((q["D"] != f["D"]).sum()); row["evsym_fixed_vs_ideal"] = int((q["E"] != i["E"]).sum())
        out["cfg"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "_comparison", "Results", "fixedpoint", "eval.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    by = load_ann()
    names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
    if a.n:
        names = names[:a.n]
    jobs = [(nm, by.get(nm, [])) for nm in names]
    t0 = time.time(); res = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap(work, jobs, chunksize=4)):
                res.append(r)
                if (k + 1) % 200 == 0:
                    print("[%d/%d] %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
    else:
        for k, j in enumerate(jobs):
            res.append(work(j))
            if (k + 1) % 50 == 0:
                print("[%d/%d] %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
    json.dump({"names": names, "configs": [c[0] for c in CONFIGS], "res": res}, open(a.out, "w"))
    print("wrote", a.out, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
