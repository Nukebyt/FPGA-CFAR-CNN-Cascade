# -*- coding: utf-8 -*-
"""Recompute the 9 side features of every candidate event as hardware-exact uint8 codes (side_fx.py) and write
Results/hw_cache_pooldetfxA/side_codes.npy (N x 9, uint8), in the event order of the existing patch cache.
usage: python extract_side_codes.py [--workers 3]"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prescreen_fx as fx
import side_fx

ROOT = fx.ROOT
OUT = os.path.join(ROOT, "_comparison", "Results", "hw_cache_pooldetfxA")
CFG = dict(sli=25, guard=17, pfa_sel=0, tau=0.6)


def scan(name):
    I = np.asarray(Image.open(os.path.join(ROOT, "HRSID", "images", name)))
    I = I[..., 0] if I.ndim == 3 else I
    h0, w0 = (I.shape[0] // 2) * 2, (I.shape[1] // 2) * 2
    I = I[:h0, :w0]
    cfg = fx.Cfg(**CFG)
    P = fx.pool_q8(I)
    g = fx.prescreen(P, cfg)
    ey, ex = np.nonzero(g["E"])
    n = len(ey)
    ev = side_fx.event_codes(g["A"][ey, ex], g["S1"][ey, ex], g["numsh"][ey, ex], P[ey, ex]) if n else np.zeros((0, 4), np.int64)
    im = side_fx.image_codes(fx.QROM[I.astype(np.int64)], I, n)
    return np.concatenate([ev, np.tile(im, (n, 1))], 1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    names = sorted(os.listdir(os.path.join(ROOT, "HRSID", "images")))
    t0 = time.time(); out = []
    import multiprocessing as mp
    with mp.Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap(scan, names, chunksize=8)):
            out.append(r)
            if (k + 1) % 500 == 0:
                print("[%d/%d] %.0fs" % (k + 1, len(names), time.time() - t0), flush=True)
    S = np.concatenate(out, 0)
    meta = np.load(os.path.join(OUT, "meta.npz"))
    assert len(S) == len(meta["labels"]), (len(S), len(meta["labels"]))
    np.save(os.path.join(OUT, "side_codes.npy"), S)
    print("wrote side_codes.npy", S.shape, "%.0fs" % (time.time() - t0))
    print("per-feature mean/std:", np.round(S.mean(0), 2), np.round(S.std(0), 2), "max", S.max(0))


if __name__ == "__main__":
    main()
