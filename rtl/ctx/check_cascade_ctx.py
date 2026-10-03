# -*- coding: utf-8 -*-
"""Independent check of a cascade_ctx_tb run: pooled store, events + per-event side codes (j, i, f0..f3), image codes f4..f8, and every CNN logit of the context network.
usage: check_cascade_ctx.py <imgw> <imgh> <imghex> <outdir> <int3.pt> <theta(negative)> <pfa> <gth> <simlog>"""
import re, sys
import numpy as np
sys.path.insert(0, r"F:\Projects\CFAR\_comparison\cnn"); sys.path.insert(0, r"F:\Projects\CFAR\_comparison\fixedpoint")
import prescreen_fx as fx, side_fx, quant_ctx as QC
W, Hh = int(sys.argv[1]), int(sys.argv[2]); imghex, outdir, ints_pt = sys.argv[3], sys.argv[4], sys.argv[5]
THETA, PFA, GTH, simlog = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
I = np.array([int(l, 16) for l in open(imghex) if l.strip()], dtype=np.int64).reshape(Hh, W)
P = fx.pool_q8(I.astype(np.uint8)); hp, wp = P.shape
store = np.array([int(l, 16) for l in open(outdir + "/rtl_store.hex") if l.strip() and not l.startswith("//")], dtype=np.int64)
ok_store = np.array_equal(store[:hp * wp].reshape(hp, wp), P)
print(f"[1] pooled store ({hp}x{wp}): {'PASS' if ok_store else 'FAIL'}")
cfg = fx.Cfg(25, 17, PFA, 0.6); assert cfg.G == GTH
g = fx.prescreen(P, cfg); ys, xs = np.nonzero(g["E"]); n = len(ys)
ev = side_fx.event_codes(g["A"][ys, xs], g["S1"][ys, xs], g["numsh"][ys, xs], P[ys, xs])
gold = [(int(y), int(x), *[int(v) for v in e]) for y, x, e in zip(ys, xs, ev)]
rtl = [tuple(int(v) for v in l.split()) for l in open(outdir + "/rtl_events.txt") if l.strip()]
ok_ev = rtl == gold
print(f"[2] events + per-event side codes: golden {n}, RTL {len(rtl)}: {'PASS' if ok_ev else 'FAIL'}")
m = re.search(r"image codes: (\d+) (\d+) (\d+) (\d+) (\d+)", open(simlog).read())
rtl_img = [int(v) for v in m.groups()]
ic = [int(v) for v in side_fx.image_codes(fx.QROM[I], I, n)]
ok_img = rtl_img == ic
print(f"[3] image codes f4..f8: RTL {rtl_img} golden {ic}: {'PASS' if ok_img else 'FAIL'}")
ints = QC.load(ints_pt)
P4 = (P[0::2, 0::2] + P[0::2, 1::2] + P[1::2, 0::2] + P[1::2, 1::2] + 2) >> 2
Pp = np.pad(P, 16, mode="edge"); P4p = np.pad(P4, 16, mode="edge")
res = [tuple(int(v) for v in l.split()) for l in open(outdir + "/rtl_results.txt") if l.strip()]
xf = np.stack([Pp[j:j + 32, i:i + 32] for j, i, *_ in gold]); xc = np.stack([P4p[(j >> 1):(j >> 1) + 32, (i >> 1):(i >> 1) + 32] for j, i, *_ in gold])
xs_ = np.array([list(e[2:]) + ic for e in gold], dtype=np.int64)
ref = QC.int_forward3(ints, xf, xc, xs_) if n else np.zeros(0, np.int64)
bad = 0
for k, (j, i, lg, acc) in enumerate(res):
    if k >= n or (j, i) != gold[k][:2] or lg != ref[k] or acc != int(ref[k] >= THETA):
        bad += 1
        if bad <= 5: print("   MISMATCH", k, (j, i, lg, acc), (gold[k][:2], ref[k]) if k < n else "")
ok_lg = bad == 0 and len(res) == n
print(f"[4] CNN logits: {len(res)} events, {bad} mismatches: {'PASS' if ok_lg else 'FAIL'}")
print("OVERALL", "PASS" if (ok_store and ok_ev and ok_img and ok_lg) else "FAIL")
