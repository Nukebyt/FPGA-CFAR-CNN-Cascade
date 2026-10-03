# -*- coding: utf-8 -*-
"""Independent reference check of a cascade_tb run.
  1. pooled store  == 2x2 round-half-up average of QROM(pixel)
  2. events        == streaming-NMS trigger + (x - c1) gate recomputed from the RTL's own detect/x/c1 stream
  3. CNN logits    == integer-exact reference on the 32x32 edge-replicated window cut from the pooled image
usage: check_cascade.py <imgw> <imgh> <imghex> <outdir> <ints.pt> <tau_q14> <theta> [TK=8]"""
import sys, math
import numpy as np
import torch
sys.path.insert(0, r"F:\Projects\CFAR\_comparison\cnn")
import quant_hw as Q

W, Hh = int(sys.argv[1]), int(sys.argv[2])
imghex, outdir, ints_pt = sys.argv[3], sys.argv[4], sys.argv[5]
TAU, THETA = int(sys.argv[6]), int(sys.argv[7])
TK = 8
X0_Q14 = 19866

pix = np.array([int(l, 16) for l in open(imghex) if l.strip()], dtype=np.int64).reshape(Hh, W)
XLO, XHI = -0.40, 2.80
qrom = np.array([math.floor(min(max((0.5 * math.log(i + 0.5) - XLO) / (XHI - XLO), 0.0), 1.0) * 255 + 0.5) for i in range(256)], dtype=np.int64)
q = qrom[pix]
P = (q[0::2, 0::2] + q[0::2, 1::2] + q[1::2, 0::2] + q[1::2, 1::2] + 2) // 4
hp, wp = P.shape

# ---- 1. pooled store
store = np.array([int(l, 16) for l in open(outdir + "/rtl_store.hex") if l.strip() and not l.startswith("//")], dtype=np.int64)
ok_store = np.array_equal(store[:hp * wp].reshape(hp, wp), P)
print(f"[1] pooled store ({hp}x{wp}): {'PASS' if ok_store else 'FAIL'}")

# ---- 2. events from the RTL detect stream
det = [tuple(int(v) for v in l.split()) for l in open(outdir + "/rtl_det.txt") if l.strip()]
Wi, Hi = W - 2 * TK, Hh - 2 * TK
assert len(det) == Wi * Hi - 1, (len(det), Wi * Hi - 1)
D = np.zeros((Hi, Wi), dtype=bool); X = np.zeros((Hi, Wi), dtype=np.int64); C1 = np.zeros((Hi, Wi), dtype=np.int64)
for k, (d, x, c1) in enumerate(det):
    m = k + 1
    D[m // Wi, m % Wi] = bool(d); X[m // Wi, m % Wi] = x; C1[m // Wi, m % Wi] = c1
ev_ref = []
for iy in range(Hi):
    for ix in range(Wi):
        if not D[iy, ix]:
            continue
        left = ix > 0 and D[iy, ix - 1]
        up = iy > 0 and D[iy - 1, ix]
        upl = iy > 0 and ix > 0 and D[iy - 1, ix - 1]
        upr = iy > 0 and ix < Wi - 1 and D[iy - 1, ix + 1]
        if left or up or upl or upr:
            continue
        g = X[iy, ix] + X0_Q14 - 2 * C1[iy, ix]
        if g >= TAU:
            ev_ref.append(((iy + TK) >> 1, (ix + TK) >> 1))
ev_rtl = [tuple(int(v) for v in l.split()) for l in open(outdir + "/rtl_events.txt") if l.strip()]
ok_ev = (ev_ref == ev_rtl)
print(f"[2] trigger+gate events: reference {len(ev_ref)}, RTL {len(ev_rtl)}: {'PASS' if ok_ev else 'FAIL'}")
if not ok_ev:
    print("   ref[:5]", ev_ref[:5], "rtl[:5]", ev_rtl[:5])
print(f"    detections in stream: {int(D.sum())}  (raw triggers before gate: see events)")

# ---- 3. CNN logits
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
    rr = np.clip(np.arange(j - 16, j + 16), 0, hp - 1)
    cc = np.clip(np.arange(i - 16, i + 16), 0, wp - 1)
    patch = P[np.ix_(rr, cc)]
    ref = int_logit(patch)
    exp_acc = int(ref >= THETA)
    if ref != logit or exp_acc != acc_flag or (j, i) != ev_rtl[n]:
        bad += 1
        if bad <= 5:
            print(f"   MISMATCH event {n} ({j},{i}): rtl logit {logit} acc {acc_flag}  ref {ref} acc {exp_acc}")
print(f"[3] CNN logits: {len(res)} events, {bad} mismatches: {'PASS' if bad == 0 and len(res) == len(ev_rtl) else 'FAIL'}")
print("OVERALL", "PASS" if (ok_store and ok_ev and bad == 0 and len(res) == len(ev_rtl)) else "FAIL")
