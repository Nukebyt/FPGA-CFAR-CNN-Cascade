# -*- coding: utf-8 -*-
"""Independent reference check of a cascade_ps_tb run (pooled prescreen + CNN).
  1. pooled store == 2x2 round-half-up average of QROM(pixel)
  2. events      == prescreen_fx.prescreen(P) (integer golden model): (j, i, A, S1, num>>12, P) in raster order
  3. CNN logits  == integer-exact reference on the 32x32 edge-replicated window cut from the pooled image
usage: check_cascade_ps.py <imgw> <imgh> <imghex> <outdir> <ints.pt> <theta> <sli> <guard> <pfa> <gth> <tau>"""
import sys
import numpy as np
import torch
sys.path.insert(0, r"F:\Projects\CFAR\_comparison\cnn")
sys.path.insert(0, r"F:\Projects\CFAR\_comparison\fixedpoint")
import quant_hw as Q
import prescreen_fx as fx

W, Hh = int(sys.argv[1]), int(sys.argv[2])
imghex, outdir, ints_pt = sys.argv[3], sys.argv[4], sys.argv[5]
THETA, SLI, GUARD, PFA, GTH, TAU = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10]), float(sys.argv[11])
pix = np.array([int(l, 16) for l in open(imghex) if l.strip()], dtype=np.int64).reshape(Hh, W)
P = fx.pool_q8(pix.astype(np.uint8)); hp, wp = P.shape
store = np.array([int(l, 16) for l in open(outdir + "/rtl_store.hex") if l.strip() and not l.startswith("//")], dtype=np.int64)
ok_store = np.array_equal(store[:hp * wp].reshape(hp, wp), P)
print(f"[1] pooled store ({hp}x{wp}): {'PASS' if ok_store else 'FAIL'}")

cfg = fx.Cfg(SLI, GUARD, PFA, TAU)
assert cfg.G == GTH, (cfg.G, GTH)
g = fx.prescreen(P, cfg)
ys, xs = np.nonzero(g["E"])
gold = [(int(y), int(x), int(g["A"][y, x]), int(g["S1"][y, x]), int(g["numsh"][y, x]), int(P[y, x])) for y, x in zip(ys, xs)]
rtl = [tuple(int(v) for v in l.split()) for l in open(outdir + "/rtl_events.txt") if l.strip()]
ok_ev = (rtl == gold)
print(f"[2] prescreen events: golden {len(gold)}, RTL {len(rtl)}: {'PASS' if ok_ev else 'FAIL'}   (D px {int(g['D'].sum())}, gated {int(g['Dg'].sum())})")

ints = torch.load(ints_pt, weights_only=False)["ints"]


def int_logit(q32):
    x = q32.astype(np.int64)[None, None]
    for L in ints:
        if L["kind"] == "conv":
            acc = Q._conv_int(x, L["W"]) + L["b"].reshape(1, -1, 1, 1)
        else:
            acc = (x.reshape(1, -1).astype(np.float64) @ L["W"].T.astype(np.float64)).round().astype(np.int64) + L["b"]
        if L["relu"]:
            M = L["M"].reshape(1, -1, *([1] * (acc.ndim - 2)))
            x = np.clip((acc * M + (1 << (L["S"] - 1))) >> L["S"], 0, 255)
        else:
            x = acc
        if L["pool"]:
            B, C, Hq, Wq = x.shape
            x = x[:, :, :Hq // 2 * 2, :Wq // 2 * 2].reshape(B, C, Hq // 2, 2, Wq // 2, 2).max(axis=(3, 5))
    return int(x.reshape(-1)[0])


res = [tuple(int(v) for v in l.split()) for l in open(outdir + "/rtl_results.txt") if l.strip()]
bad = 0
for n, (j, i, logit, acc_flag) in enumerate(res):
    rr = np.clip(np.arange(j - 16, j + 16), 0, hp - 1); cc = np.clip(np.arange(i - 16, i + 16), 0, wp - 1)
    ref = int_logit(P[np.ix_(rr, cc)]); exp_acc = int(ref >= THETA)
    if ref != logit or exp_acc != acc_flag or (j, i) != tuple(rtl[n][:2]):
        bad += 1
        if bad <= 5:
            print(f"   MISMATCH event {n} ({j},{i}): rtl logit {logit} acc {acc_flag}  ref {ref} acc {exp_acc}")
ok_log = (bad == 0 and len(res) == len(rtl))
print(f"[3] CNN logits: {len(res)} events, {bad} mismatches: {'PASS' if ok_log else 'FAIL'}")
print("OVERALL", "PASS" if (ok_store and ok_ev and ok_log) else "FAIL")
