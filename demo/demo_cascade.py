# -*- coding: utf-8 -*-
"""Demo: run one HRSID image through the Weibull-CFAR + INT8-CNN cascade and show the result.
The HRSID dataset is expected in <repo>/HRSID (images/ and annotations/), or pass --hrsid <folder>.

  python demo_cascade.py --image 1234                 # on the DE10-Standard board (bitstream cascade_ctx_jtag_top.sof)
  python demo_cascade.py --image 1234 --sim           # software only (bit-exact golden models), no board needed
  python demo_cascade.py --image P0068_600_1400_4200_5000.png --theta -10695

What it does
  1. loads the image (and its ship annotations, if any)
  2. board mode: uploads it over JTAG and reads back every event, integer logit, accept flag and the on-chip cycle counters;
     sim mode: computes the same with the Python golden models
  3. recomputes the frame with the golden models and reports whether the board matched them exactly (board mode)
  4. draws the prescreen events and the accepted detections over the image and saves/shows the figure

Thresholds (integer logit, set on validation images): 90 % -5979, 95 % -10695, 97 % -14462 (default), 98 % -17056.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "_comparison", "fixedpoint"))
sys.path.insert(0, os.path.join(ROOT, "_comparison", "cnn"))
sys.path.insert(0, os.path.join(ROOT, "_comparison", "cnn", "sweep250", "board"))
import prescreen_fx as fx      # noqa: E402
import quant_ctx as QC         # noqa: E402
import side_fx                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", default="P0068_600_1400_4200_5000.png", help="HRSID file name or index")
ap.add_argument("--theta", type=int, default=-14462, help="CNN threshold on the integer logit (97 %% retention by default)")
ap.add_argument("--pfa", type=int, default=0, help="0: 3e-2 (deployed)  1: 1e-2  2: 1e-3  3: 1e-4")
ap.add_argument("--sim", action="store_true", help="software only, no board")
ap.add_argument("--out", default=os.path.join(HERE, "out"))
ap.add_argument("--no-show", action="store_true")
ap.add_argument("--hrsid", default=os.path.join(ROOT, "HRSID"), help="HRSID folder (with images/ and annotations/)")
a = ap.parse_args()

names = sorted(os.listdir(os.path.join(a.hrsid, "images")))
name = names[int(a.image)] if a.image.isdigit() else a.image
img = np.asarray(Image.open(os.path.join(a.hrsid, "images", name)))
img = (img[..., 0] if img.ndim == 3 else img).astype(np.uint8)
cfg = fx.Cfg(25, 17, a.pfa, 0.6)
ints = QC.load(os.path.join(ROOT, "_comparison", "Results", "hw", "pf_full_q8_int3.pt"))


def golden(img):
    """Python golden models: events (pooled j, i) and integer logits, exactly as the RTL computes them."""
    P = fx.pool_q8(img); g = fx.prescreen(P, cfg)
    ys, xs = np.nonzero(g["E"]); n = len(ys)
    if n == 0:
        return ys, xs, np.zeros(0, np.int64)
    ic = side_fx.image_codes(fx.QROM[img.astype(np.int64)], img, n)
    codes = side_fx.event_codes(g["A"][ys, xs], g["S1"][ys, xs], g["numsh"][ys, xs], P[ys, xs])
    P4 = (P[0::2, 0::2] + P[0::2, 1::2] + P[1::2, 0::2] + P[1::2, 1::2] + 2) >> 2
    Pp = np.pad(P, 16, mode="edge"); P4p = np.pad(P4, 16, mode="edge")
    out = []
    for k0 in range(0, n, 256):
        sl = slice(k0, k0 + 256)
        xf = np.stack([Pp[j:j + 32, i:i + 32] for j, i in zip(ys[sl], xs[sl])])
        xc = np.stack([P4p[(j >> 1):(j >> 1) + 32, (i >> 1):(i >> 1) + 32] for j, i in zip(ys[sl], xs[sl])])
        xsd = np.concatenate([codes[sl], np.tile(ic, (len(codes[sl]), 1))], 1)
        out.append(QC.int_forward3(ints, xf, xc, xsd))
    return ys, xs, np.concatenate(out)


print(f"image {name}  ({img.shape[1]}x{img.shape[0]})")
t0 = time.time()
gy, gx, glog = golden(img)
t_sw = time.time() - t0

if a.sim:
    j, i, logit = gy, gx, glog
    print(f"[software] {len(j)} prescreen events, {int((logit >= a.theta).sum())} accepted   ({t_sw:.1f} s in Python)")
else:
    from board_ctx import BoardCTX
    b = BoardCTX()
    r = b.run_frame(img, pfa_sel=a.pfa, g_th=cfg.G, theta=a.theta)
    b.close()
    j, i, logit = r["j"], r["i"], r["logit"]
    print(f"[board]    {r['n_res']} prescreen events, {r['n_accepted']} accepted")
    print(f"[board]    prescreen pass {r['ps_cycles'] * 1e-5:.2f} ms, after last pixel {r['post_cycles'] * 2e-5:.2f} ms "
          f"(on-chip counters; JTAG upload {r['t_upload_s']:.2f} s is the test set-up only)")
    same_ev = len(j) == len(gy) and np.array_equal(j, gy) and np.array_equal(i, gx)
    same_lg = same_ev and np.array_equal(logit, glog) and np.array_equal(r["accept"], (glog >= a.theta).astype(int))
    print(f"[check]    board vs golden model: events {'IDENTICAL' if same_ev else 'DIFFERENT'}, "
          f"logits and accept flags {'IDENTICAL' if same_lg else 'DIFFERENT'}")

acc = logit >= a.theta
ys, xs = 2 * np.asarray(j) + 1, 2 * np.asarray(i) + 1          # pooled -> full-resolution position

# ship annotations (green outlines) and a simple found/false count with the report's 4-pixel rule
ann = json.load(open(os.path.join(a.hrsid, "annotations", "train_test2017.json")))
iid = [im["id"] for im in ann["images"] if im["file_name"] == name]
polys = []
for an in ann["annotations"]:
    if iid and an["image_id"] == iid[0]:
        sg = an["segmentation"][0] if isinstance(an["segmentation"][0], list) else an["segmentation"]
        polys.append([(sg[k], sg[k + 1]) for k in range(0, len(sg) - 1, 2)])
found = false = 0
if polys:
    from scipy import ndimage
    near_any = np.zeros(len(ys), bool)
    for p in polys:
        m = Image.new("L", (img.shape[1], img.shape[0]), 0); ImageDraw.Draw(m).polygon(p, fill=1)
        d = ndimage.distance_transform_edt(~np.array(m, bool))
        near = d[ys, xs] <= 4
        near_any |= near
        found += int((near & acc).any())
    false = int((acc & ~near_any).sum())
    print(f"[result]   {found} of {len(polys)} annotated ships found, {false} false detections (threshold {a.theta})")

import matplotlib
if a.no_show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 6.8))
for k in range(2):
    ax[k].imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
    for p in polys:
        q = np.array(p + [p[0]]); ax[k].plot(q[:, 0], q[:, 1], color="#1baf7a", lw=1.3)
    ax[k].set_xticks([]); ax[k].set_yticks([])
ax[0].scatter(xs, ys, s=12, facecolors="none", edgecolors="#eda100", lw=0.8)
ax[0].set_title(f"Weibull prescreen: {len(ys)} candidate events", loc="left")
ax[1].scatter(xs[acc], ys[acc], s=70, facecolors="none", edgecolors="#e34948", lw=1.6)
ax[1].set_title(f"after the INT8 CNN: {int(acc.sum())} accepted" + (f", {found}/{len(polys)} ships, {false} false" if polys else ""), loc="left")
fig.suptitle(f"{name}  ({'software model' if a.sim else 'read back from the FPGA'})", fontsize=11)
fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
png = os.path.join(a.out, os.path.splitext(name)[0] + ("_sim" if a.sim else "_board") + ".png")
fig.savefig(png, dpi=120)
print("saved", png)
if not a.no_show:
    plt.show()
