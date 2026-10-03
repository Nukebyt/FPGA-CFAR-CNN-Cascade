# -*- coding: utf-8 -*-
"""Step 2 (simple, validation-tuned): s' = s + alpha * clip(neighbour evidence),  neighbour evidence = mean of the top-N scores of the OTHER events within R px.
alpha, R, N chosen on the validation images to minimise FA/img at 97 % ship retention; applied unchanged to test.
usage: HWDATA=pooldetA python fuse_simple.py <run_name>     -> Results/hw/<run_name>_fs_{val,test}.npy"""
import os, sys, itertools, numpy as np
from scipy.spatial import cKDTree
import hwlib as H
name = sys.argv[1]
d = H.HWData(); m = np.load(os.path.join(H.CACHE, "meta.npz")); cxy = np.array(m["cxy"]).T; gt2 = m["gtIdx2"].astype(np.int64)
sv = np.load(os.path.join(H.RES, "hw", f"{name}_val.npy")).astype(np.float64); st = np.load(os.path.join(H.RES, "hw", f"{name}_test.npy")).astype(np.float64)
vi, ti = d.idx["val"], d.idx["test"]
nv, nt = len(d.split_imgs["val"]), len(d.split_imgs["test"])


def neigh(idx, sc, R, N):
    img = d.img[idx]; out = np.full(len(idx), -8.0)
    o = np.argsort(img, kind="stable"); st_ = np.r_[0, np.nonzero(np.diff(img[o]))[0] + 1, len(o)]
    for a, b in zip(st_[:-1], st_[1:]):
        ii = o[a:b]; p = cxy[idx[ii]]; s = sc[ii]; tree = cKDTree(p)
        for j, lst in enumerate(tree.query_ball_point(p, R)):
            lst = [q for q in lst if q != j]
            if lst: out[ii[j]] = np.sort(np.clip(s[lst], -8, 8))[::-1][:N].mean()
    return out


def fa97(idx, sc, nimg, r=0.97):
    img, g1, g2, lab = d.img[idx], d.gt[idx], gt2[idx], d.labels[idx]
    k = np.concatenate([(img * 256 + g1)[lab & (g1 > 0)], (img * 256 + g2)[g2 > 0]]); s = np.concatenate([sc[lab & (g1 > 0)], sc[g2 > 0]])
    o = np.argsort(k, kind="stable"); k, s = k[o], s[o]; u, f = np.unique(k, return_index=True); sm = np.maximum.reduceat(s, f)
    thr = np.sort(sm)[int(np.floor((1 - r) * len(sm)))]
    return (sc[~lab] >= thr).sum() / nimg


base = fa97(vi, sv, nv); print(f"val FA/img @97 % without fusion: {base:.2f}")
best = (base, None); cache = {}
for R, N in itertools.product((16, 32, 64), (1, 2)):
    nb = neigh(vi, sv, R, N); cache[(R, N)] = nb
    for al in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        v = fa97(vi, sv + al * nb, nv)
        if v < best[0]: best = (v, (R, N, al))
    print(f"  R={R} N={N}: best over alpha so far {best[0]:.2f} {best[1]}", flush=True)
v, (R, N, al) = best if best[1] else (base, (16, 1, 0.0))
print(f"chosen R={R} N={N} alpha={al}: val FA@97 {v:.2f} (was {base:.2f})")
np.save(os.path.join(H.RES, "hw", f"{name}_fs_val.npy"), sv + al * cache.get((R, N), neigh(vi, sv, R, N)))
np.save(os.path.join(H.RES, "hw", f"{name}_fs_test.npy"), st + al * neigh(ti, st, R, N))
