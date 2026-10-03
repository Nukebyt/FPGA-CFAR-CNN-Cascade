# -*- coding: utf-8 -*-
"""On-board sweep of the pooled-prescreen cascade (cascade_ctx_jtag_top) with an independent bit-exact check of every frame.
For each image: upload to the board, read the events + INT8 logits back, recompute the same frame with the Python golden models
(prescreen_fx.prescreen on the pooled 8-bit frame, then the integer CNN forward of pf_plain_q8_int.pt on every event patch) and compare.
usage: python board_sweep_ps.py --n 250 --first 0 --pfa 0 --theta -17457 [--out Results/sweep_ps]
The board must be powered, connected by USB-Blaster and programmed with _quartus/cascade_ps_jtag/output_files/cascade_ctx_jtag_top.sof."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "_comparison", "fixedpoint"))
sys.path.insert(0, os.path.join(ROOT, "_comparison", "cnn"))
import prescreen_fx as fx                                          # noqa: E402
import quant_ctx as QC                                             # noqa: E402
import side_fx                                                     # noqa: E402
from board_ctx import BoardCTX                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=250); ap.add_argument("--first", type=int, default=0)
ap.add_argument("--pfa", type=int, default=0); ap.add_argument("--theta", type=int, default=-14462)
ap.add_argument("--tau", type=float, default=0.6); ap.add_argument("--out", default=os.path.join(ROOT, "_comparison", "Results", "sweep_ctx"))
ap.add_argument("--ints", default=os.path.join(ROOT, "_comparison", "Results", "hw", "pf_full_q8_int3.pt"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cfg = fx.Cfg(25, 17, a.pfa, a.tau)
ints = QC.load(a.ints)
names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
print("G =", cfg.G, " KC =", cfg.KC, flush=True)


def golden(img, P, g, ev):
    """events, side codes, image codes and integer logits of the Python models for this frame"""
    ys = np.array([e[0] for e in ev], np.int64); xs = np.array([e[1] for e in ev], np.int64)
    n = len(ev)
    ic = side_fx.image_codes(fx.QROM[img.astype(np.int64)], img, n)
    if n == 0:
        return np.zeros(0, np.int64)
    codes = side_fx.event_codes(g["A"][ys, xs], g["S1"][ys, xs], g["numsh"][ys, xs], P[ys, xs])
    P4 = (P[0::2, 0::2] + P[0::2, 1::2] + P[1::2, 0::2] + P[1::2, 1::2] + 2) >> 2
    Pp = np.pad(P, 16, mode="edge"); P4p = np.pad(P4, 16, mode="edge")
    out = []
    for k0 in range(0, n, 256):
        sl = slice(k0, k0 + 256)
        xf = np.stack([Pp[j:j + 32, i:i + 32] for j, i in zip(ys[sl], xs[sl])]); xc = np.stack([P4p[(j >> 1):(j >> 1) + 32, (i >> 1):(i >> 1) + 32] for j, i in zip(ys[sl], xs[sl])])
        xsd = np.concatenate([codes[sl], np.tile(ic, (len(codes[sl]), 1))], 1)
        out.append(QC.int_forward3(ints, xf, xc, xsd))
    return np.concatenate(out)


b = BoardCTX()
print("board:", b.status(), flush=True)
tot_ev = tot_bad = 0; t0 = time.time(); log = []
for k in range(a.first, min(a.first + a.n, len(names))):
    img = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", names[k]))); img = (img[..., 0] if img.ndim == 3 else img).astype(np.uint8)
    r = b.run_frame(img, pfa_sel=a.pfa, g_th=cfg.G, theta=a.theta)
    P = fx.pool_q8(img); g = fx.prescreen(P, cfg)
    ys, xs = np.nonzero(g["E"]); ev = list(zip(ys.tolist(), xs.tolist()))
    lg = golden(img, P, g, ev)
    ok_ev = (r["n_res"] == len(ev) and list(zip(r["j"].tolist(), r["i"].tolist())) == ev)
    ok_lg = ok_ev and np.array_equal(r["logit"], lg) and np.array_equal(r["accept"], (lg >= a.theta).astype(int))
    bad = 0 if (ok_ev and ok_lg and not r["status"]["ev_overflow"]) else 1
    tot_ev += len(ev); tot_bad += bad
    log.append(dict(k=k, name=names[k], events=len(ev), n_rtl=int(r["n_res"]), ok=not bad, ps_cycles=int(r["ps_cycles"]), post_cycles=int(r["post_cycles"]), upload_s=r["t_upload_s"]))
    np.savez(os.path.join(a.out, f"frame_{k:04d}.npz"), j=r["j"], i=r["i"], logit=r["logit"], accept=r["accept"], name=names[k])
    print(f"img {k:4d} {len(ev):4d} events  {'OK' if not bad else 'MISMATCH'}  prescreen {r['ps_cycles']*10e-6:5.2f} ms  post {r['post_cycles']*20e-6:7.2f} ms  upload {r['t_upload_s']:5.1f}s  | {time.time()-t0:.0f}s", flush=True)
json.dump(log, open(os.path.join(a.out, "sweep_log.json"), "w"))
print(f"DONE: {len(log)} frames, {tot_ev} events, {tot_bad} frames with any mismatch")
b.close()
