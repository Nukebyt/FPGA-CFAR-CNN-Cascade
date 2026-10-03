# -*- coding: utf-8 -*-
"""Hardware-exact candidate set for the cascade CNN: every contrast-peak event of the INTEGER prescreen (prescreen_fx), config A
(pooled 2x2, window 25/17, Pfa 3e-2, gate 0.6), over all 5,604 HRSID images, with
   patches : 32x32 window of the pooled 8-bit store P around the event (edge replicate)          (what patch_fetch.v reads)
   ctx     : 32x32 window of P4 = 2x2-pool of P (a second 200x200 store), centred on (j>>1, i>>1)  (context tower input)
   side    : 9 scalars [contrast (ln units), local mean (ln), local sigma (ln), event amplitude (ln),
                        image mean, image s.d. of the ln-amplitude, fraction >200, fraction <5, ln(1+n_events)]
Writes Results/hw_cache_pooldetfxA/{patches.npy, ctx.npy, side.npy, meta.npz} in the layout hwlib.HWData / train_ctx expect.
usage: python extract_fx_events.py [--workers 3] [--n N]"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prescreen_fx as fx
from eval_fx_dataset import load_ann, ship_masks

ROOT = fx.ROOT
OUT = os.path.join(ROOT, "_comparison", "Results", "hw_cache_pooldetfxA")
TOL = 4
CFG = dict(sli=25, guard=17, pfa_sel=0, tau=0.6)


def pool2(P):
    h, w = (P.shape[0] // 2) * 2, (P.shape[1] // 2) * 2
    return (P[0:h:2, 0:w:2] + P[0:h:2, 1:w:2] + P[1:h:2, 0:w:2] + P[1:h:2, 1:w:2] + 2) >> 2


def scan(args):
    k, name, anns = args
    I = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", name)))
    I = I[..., 0] if I.ndim == 3 else I
    h0, w0 = (I.shape[0] // 2) * 2, (I.shape[1] // 2) * 2
    I = I[:h0, :w0]
    masks = ship_masks(anns, h0, w0)
    dists = [ndimage.distance_transform_edt(~m) for m in masks]
    cfg = fx.Cfg(**CFG)
    P = fx.pool_q8(I)
    g = fx.prescreen(P, cfg)
    ey, ex = np.nonzero(g["E"])
    n = len(ey)
    ry, rx = 2 * ey + 1, 2 * ex + 1
    gt = np.zeros(n, np.uint8); gt2 = np.zeros(n, np.uint8)
    for s, d in enumerate(dists, 1):
        hit = d[ry, rx] <= TOL
        gt2[hit & (gt != 0) & (gt2 == 0)] = s
        gt[hit & (gt == 0)] = s
    N = cfg.N
    A = g["A"][ey, ex].astype(np.float64); S1 = g["S1"][ey, ex].astype(np.float64); num = g["num"][ey, ex].astype(np.float64)
    st = fx.STEP
    q = fx.QROM[I.astype(np.int64)].astype(np.float64)
    qm, qs = q.mean(), q.std()
    side = np.zeros((n, 9), np.float32)
    side[:, 0] = A / N * st
    side[:, 1] = fx.XLO + st * S1 / N
    side[:, 2] = st * np.sqrt(num / (N * (N - 1)))
    side[:, 3] = fx.XLO + st * P[ey, ex]
    side[:, 4] = fx.XLO + st * qm; side[:, 5] = st * qs
    side[:, 6] = (I > 200).mean(); side[:, 7] = (I < 5).mean(); side[:, 8] = np.log(1 + n)
    return dict(k=k, ey=ey.astype(np.int16), ex=ex.astype(np.int16), gt=gt, gt2=gt2, side=side, nShips=len(masks), gate=(A / N * st).astype(np.float32),
                c1=side[:, 1].copy(), xt=side[:, 3].copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    by = load_ann()
    names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
    if a.n:
        names = names[:a.n]
    jobs = [(k, nm, by.get(nm, [])) for k, nm in enumerate(names)]
    t0 = time.time(); res = []
    import multiprocessing as mp
    with mp.Pool(a.workers) as pool:
        for r in pool.imap(scan, jobs, chunksize=4):
            res.append(r)
            if len(res) % 300 == 0:
                print("[scan %d/%d] %.0fs  events so far %d" % (len(res), len(jobs), time.time() - t0, sum(len(x["ey"]) for x in res)), flush=True)
    res.sort(key=lambda r: r["k"])
    nev = sum(len(r["ey"]) for r in res)
    print("events", nev, "(%.1f/img)" % (nev / len(res)), flush=True)
    pat = np.lib.format.open_memmap(os.path.join(OUT, "patches.npy"), mode="w+", dtype=np.uint8, shape=(nev, 32, 32))
    ctx = np.lib.format.open_memmap(os.path.join(OUT, "ctx.npy"), mode="w+", dtype=np.uint8, shape=(nev, 32, 32))
    o = 0
    for r, (k, nm, an) in zip(res, jobs):
        n = len(r["ey"])
        if n:
            I = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", nm)))
            I = I[..., 0] if I.ndim == 3 else I
            P = fx.pool_q8(I[: (I.shape[0] // 2) * 2, : (I.shape[1] // 2) * 2])
            Pp = np.pad(P, 16, mode="edge").astype(np.uint8)
            P4 = pool2(P); P4p = np.pad(P4, 16, mode="edge").astype(np.uint8)
            for t in range(n):
                j, i = int(r["ey"][t]), int(r["ex"][t])
                pat[o + t] = Pp[j:j + 32, i:i + 32]
                jj, ii = min(j >> 1, P4.shape[0] - 1), min(i >> 1, P4.shape[1] - 1)
                ctx[o + t] = P4p[jj:jj + 32, ii:ii + 32]
        o += n
    pat.flush(); ctx.flush()
    cat = lambda k: np.concatenate([r[k] for r in res]) if nev else np.zeros(0)
    gt, gt2 = cat("gt"), cat("gt2")
    side = np.concatenate([r["side"] for r in res], 0)
    img = np.concatenate([np.full(len(r["ey"]), r["k"] + 1, np.uint16) for r in res])
    cxy = np.stack([np.concatenate([2 * r["ey"].astype(np.int32) + 1 for r in res]), np.concatenate([2 * r["ex"].astype(np.int32) + 1 for r in res])], 0).astype(np.float32)
    z = np.zeros(nev, np.float32)
    np.save(os.path.join(OUT, "side.npy"), side)
    np.savez(os.path.join(OUT, "meta.npz"), gtIdx2=gt2, labels=(gt > 0), imgIdx=img, gtIdx=gt, area=z, bboxH=z, bboxW=z, peakX=cat("xt"), meanX=cat("xt"),
             c1=cat("c1"), c2=z, imgNShips=np.array([r["nShips"] for r in res]), cxy=cxy, gate3=cat("gate"), imgNTrig=np.array([len(r["ey"]) for r in res]))
    print("wrote", OUT, "%.0fs" % (time.time() - t0), " on-ship events %.2f%%" % (100 * (gt > 0).mean()))


if __name__ == "__main__":
    main()
