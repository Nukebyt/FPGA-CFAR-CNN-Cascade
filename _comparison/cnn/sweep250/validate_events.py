# -*- coding: utf-8 -*-
"""Sanity check: plane-0 events rebuilt in sweep250.py must equal the earlier MATLAB hwspec extraction
(Results/cnn_patches_hwexact_test.mat): same trigger positions, same gate decision, same 32x32 patches."""
import os, sys, numpy as np, h5py
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep250 as S
names, Hh, Ww, Didx, Dg, gts = S.load_maps()
imgl = np.loadtxt(os.path.join(S.OUT, "img_list.txt")).astype(int)
f = h5py.File(os.path.join(S.RES, "cnn_patches_hwexact_test.mat"), "r")
pat = f["patches"]; img = np.array(f["imgIdx"]).ravel().astype(int); tyx = np.array(f["tyx"]); gate = np.array(f["gate"]).ravel()
print("mat patches h5 shape", pat.shape, "tyx", tyx.shape)
if tyx.shape[0] == 2: tyx = tyx.T
tot_m = tot_mine = same = 0; pe_n = pe_t = 0
for k in range(0, len(imgl), 5):
    h, w = int(Hh[k]), int(Ww[k])
    I = np.asarray(Image.open(os.path.join(S.IMG_DIR, names[k])), dtype=np.float64)
    I = (I[:, :, 0] if I.ndim == 3 else I)[:h, :w]
    P = S.pooled_store(I)
    idx = Didx[k][0]; y, x = idx % h, idx // h
    D = np.zeros((h, w), bool); D[y, x] = True
    ty, tx = np.nonzero(S.triggers(D))
    gq = dict(zip(idx.tolist(), Dg[k][0].tolist()))
    g = np.array([gq[int(a + b * h)] for a, b in zip(ty, tx)]) / 16384.0
    ok = g >= S.GATE_TAU
    mine = {(int(a) + 1, int(b) + 1) for a, b in zip(ty[ok], tx[ok])}
    sel = np.nonzero((img == imgl[k]) & (gate >= S.GATE_TAU))[0]
    ref = {(int(tyx[i, 0]), int(tyx[i, 1])) for i in sel}
    tot_m += len(ref); tot_mine += len(mine); same += len(ref & mine)
    for i in sel[:3]:
        a, b = int(tyx[i, 0]) - 1, int(tyx[i, 1]) - 1
        mp = S.patch_at(P, a, b); rp = np.array(pat[i])
        pe_t += 1
        if np.array_equal(mp, rp): pe_n += 1
        elif np.array_equal(mp.T, rp): pe_n += 1000
print(f"gated events: mat {tot_m}, rebuilt {tot_mine}, identical positions {same}")
print(f"patch compare ({pe_t} patches): direct-equal count + 1000*transposed-equal count = {pe_n}")
