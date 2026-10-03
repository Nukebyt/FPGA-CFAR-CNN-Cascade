# -*- coding: utf-8 -*-
"""Stream the sweep images through the cascade ON THE FPGA and store the hardware results.

For every image of Results/sweep250/img_list.txt and every Pfa plane the board returns the complete candidate-event list
(position + integer CNN logit of EVERY gated event: theta is set to the minimum so nothing is rejected on chip; the accept
decision at any operating point is then logit >= theta, applied offline in sweep250.py --hw).
Resumable: frame_<img>_<plane>.npz files that already exist are skipped.

usage:  python board_sweep.py [--n 250] [--planes 0 1 2 3] [--first 0]
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import Board                                          # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = os.path.join(ROOT, "_comparison", "Results", os.environ.get("SWEEP", "sweep250"))
HW = os.path.join(OUT, "hw")
IMG_DIR = os.path.join(ROOT, "HRSID", "images")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--planes", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--tau", type=int, default=12288)             # 0.75 in Q.14
a = ap.parse_args()

os.makedirs(HW, exist_ok=True)
import h5py                                                    # names live in det_maps.mat (written by sweep250_extract.m)
f = h5py.File(os.path.join(OUT, "det_maps.mat"), "r")
names = ["".join(chr(c) for c in f[f["names"][0, k]][()].ravel()) for k in range(f["H"].shape[1])]

b = Board()
print("board status:", b.status(), flush=True)
t0 = time.time(); done = 0
for k in range(a.first, min(a.first + a.n, len(names))):
    todo = [p for p in a.planes if not os.path.exists(os.path.join(HW, f"frame_{k:03d}_{p}.npz"))]
    if not todo:
        continue
    img = np.asarray(Image.open(os.path.join(IMG_DIR, names[k])))
    img = (img[:, :, 0] if img.ndim == 3 else img).astype(np.uint8)
    for p in todo:
        r = b.run_frame(img, pfa_sel=p, tau_q14=a.tau, theta=-(2 ** 31) + 1)
        assert r["pix_issued"] == 800 * 800 and not r["status"]["ev_overflow"] and not r["status"]["dropped"], r
        np.savez(os.path.join(HW, f"frame_{k:03d}_{p}.npz"), j=r["j"], i=r["i"], logit=r["logit"], accept=r["accept"],
                 n_events=r["n_events"], post_cycles=r["post_cycles"], t_upload_s=r["t_upload_s"], name=names[k], plane=p)
        done += 1
        print(f"img {k:3d} plane {p}: {r['n_events']:4d} events  upload {r['t_upload_s']:5.1f}s  post {r['post_cycles']*20e-6:6.1f} ms "
              f"| {done} frames, {time.time()-t0:.0f}s elapsed", flush=True)
b.close()
