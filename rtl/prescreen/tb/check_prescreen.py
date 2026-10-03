# -*- coding: utf-8 -*-
"""Run prescreen_tb.v on a pooled crop and compare every event (j, i, A, S1, num>>12, P) with the Python golden model.
usage: python check_prescreen.py --img NAME --y0 0 --x0 0 --h 128 --w 128 --sli 25 --guard 17 --pfa 0 --tau 0.6"""
import argparse, os, subprocess, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "_comparison", "fixedpoint"))
import prescreen_fx as fx

ap = argparse.ArgumentParser()
ap.add_argument("--img", default="0.png"); ap.add_argument("--y0", type=int, default=0); ap.add_argument("--x0", type=int, default=0)
ap.add_argument("--h", type=int, default=128); ap.add_argument("--w", type=int, default=128)
ap.add_argument("--sli", type=int, default=25); ap.add_argument("--guard", type=int, default=17)
ap.add_argument("--pfa", type=int, default=0); ap.add_argument("--tau", type=float, default=0.6)
ap.add_argument("--out", default=os.path.join(HERE, "out")); ap.add_argument("--noise", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.noise:
    rng = np.random.RandomState(a.noise); I = (rng.rayleigh(30, (a.h, a.w))).clip(0, 255).astype(np.uint8)
    I[40:52, 60:80] = 220; I[8:12, 10:14] = 255
else:
    I = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", a.img)))
    I = I[..., 0] if I.ndim == 3 else I
    I = I[a.y0:a.y0 + a.h, a.x0:a.x0 + a.w]
P = fx.pool_q8(I); HP, WP = P.shape
cfg = fx.Cfg(a.sli, a.guard, a.pfa, a.tau)
g = fx.prescreen(P, cfg)
ys, xs = np.nonzero(g["E"])
gold = [(int(y), int(x), int(g["A"][y, x]), int(g["S1"][y, x]), int(g["numsh"][y, x]), int(P[y, x])) for y, x in zip(ys, xs)]
np.savetxt(os.path.join(a.out, "P.hex"), P.reshape(-1), fmt="%02x")
kc = [fx.Cfg(a.sli, a.guard, p, a.tau).KC for p in range(4)]
defs = ["-DWP=%d" % WP, "-DHP=%d" % HP, "-DSLI=%d" % a.sli, "-DGUARD=%d" % a.guard, "-DKC0=%d" % kc[0], "-DKC1=%d" % kc[1], "-DKC2=%d" % kc[2], "-DKC3=%d" % kc[3],
        "-DNUM_LO=%d" % cfg.NUM_LO, "-DNUM_HI=%d" % cfg.NUM_HI, "-DPFA=%d" % a.pfa, "-DMAXCYC=%d" % (((HP + 2*((a.sli-1)//2))*(WP + 2*((a.sli-1)//2))*5 + 3*(WP+2)*2 + 5000)), "-DGTH=%d" % cfg.G,
        '-DPHEX="%s"' % os.path.join(a.out, "P.hex").replace("\\", "/"), '-DOUTFILE="%s"' % os.path.join(a.out, "rtl_events.txt").replace("\\", "/"), '-DOUTFILE2="%s"' % os.path.join(a.out, "rtl_events2.txt").replace("\\", "/")]
vvp = os.path.join(a.out, "prescreen.vvp")
subprocess.check_call(["iverilog", "-g2005-sv", "-o", vvp] + defs + [os.path.join(HERE, "prescreen_tb.v"), os.path.join(HERE, "..", "prescreen_top.v"),
                      os.path.join(HERE, "..", "prescreen_core.v"), os.path.join(HERE, "..", "peak5.v")])
r = subprocess.run(["vvp", vvp], capture_output=True, text=True); print(r.stdout.strip()[-300:])
rtl = [tuple(int(v) for v in ln.split()) for ln in open(os.path.join(a.out, "rtl_events.txt")) if ln.strip()]
print("pooled %dx%d  golden events %d  RTL events %d  (D px %d, gated %d)" % (WP, HP, len(gold), len(rtl), g["D"].sum(), g["Dg"].sum()))
rtl2 = [tuple(int(v) for v in ln.split()) for ln in open(os.path.join(a.out, "rtl_events2.txt")) if ln.strip()]
print("second back-to-back pass equal to first:", rtl2 == rtl)
ok = rtl == gold and rtl2 == rtl
print("BIT-EXACT MATCH" if ok else "MISMATCH")
if not ok:
    sg, sr = set(gold), set(rtl)
    print(" only golden:", sorted(sg - sr)[:8]); print(" only RTL   :", sorted(sr - sg)[:8])
    print(" order equal:", [e for e in rtl if e in sg] == [e for e in gold if e in sr])
sys.exit(0 if ok else 1)
