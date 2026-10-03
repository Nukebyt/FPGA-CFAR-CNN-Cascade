# -*- coding: utf-8 -*-
"""On-board sweep of the pooled-prescreen cascade (cascade_ps_jtag_top) with an independent bit-exact check of every frame.
For each image: upload to the board, read the events + INT8 logits back, recompute the same frame with the Python golden models
(prescreen_fx.prescreen on the pooled 8-bit frame, then the integer CNN forward of pf_plain_q8_int.pt on every event patch) and compare.
usage: python board_sweep_ps.py --n 250 --first 0 --pfa 0 --theta -17457 [--out Results/sweep_ps]
The board must be powered, connected by USB-Blaster and programmed with _quartus/cascade_ps_jtag/output_files/cascade_ps_jtag_top.sof."""
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
import quant_hw as Q                                               # noqa: E402
from board_ps import BoardPS                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=250); ap.add_argument("--first", type=int, default=0)
ap.add_argument("--pfa", type=int, default=0); ap.add_argument("--theta", type=int, default=-17457)
ap.add_argument("--tau", type=float, default=0.6); ap.add_argument("--out", default=os.path.join(ROOT, "_comparison", "Results", "sweep_ps"))
ap.add_argument("--ints", default=os.path.join(ROOT, "_comparison", "Results", "hw", "pf_plain_q8_int.pt"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cfg = fx.Cfg(25, 17, a.pfa, a.tau)
ints = torch.load(a.ints, weights_only=False)["ints"]
names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
print("G =", cfg.G, " KC =", cfg.KC, flush=True)


def logits(P, ev):
    Pp = np.pad(P, 16, mode="edge")
    patches = np.stack([Pp[j:j + 32, i:i + 32] for j, i in ev]).astype(np.int64) if len(ev) else np.zeros((0, 32, 32), np.int64)
    out = []
    for k in range(0, len(patches), 512):
        out.append(Q.int_forward(ints, patches[k:k + 512]) / (ints[-1]["s_in"] * ints[-1]["sw"][0]))
    return np.rint(np.concatenate(out)).astype(np.int64) if out else np.zeros(0, np.int64)


b = BoardPS()
print("board:", b.status(), flush=True)
tot_ev = tot_bad = 0; t0 = time.time(); log = []
for k in range(a.first, min(a.first + a.n, len(names))):
    img = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", names[k]))); img = (img[..., 0] if img.ndim == 3 else img).astype(np.uint8)
    r = b.run_frame(img, pfa_sel=a.pfa, g_th=cfg.G, theta=a.theta)
    P = fx.pool_q8(img); g = fx.prescreen(P, cfg)
    ys, xs = np.nonzero(g["E"]); ev = list(zip(ys.tolist(), xs.tolist()))
    lg = logits(P, ev)
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
