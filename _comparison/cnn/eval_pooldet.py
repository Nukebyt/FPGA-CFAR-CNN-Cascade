# -*- coding: utf-8 -*-
"""End-to-end cascade numbers for a pooled-domain-prescreen dataset.
usage:  HWDATA=pooldetA python eval_pooldet.py <run_name> [suffix]        (scores: Results/hw/<run_name><suffix>_{val,test}.npy)
A ship is delivered by the prescreen if >=1 event lies within 4 px of its polygon (events touching two ships count for both), and it survives the CNN if one of those
events is accepted.  Thresholds are picked on the VALIDATION images for a target retention and applied to the TEST images.
cascade Pd = ships with an accepted on-ship event / ALL ships;  FA = accepted events on no ship, per image."""
import os, sys, numpy as np
import hwlib as H
name = sys.argv[1]; suf = sys.argv[2] if len(sys.argv) > 2 else ""
d = H.HWData()
m = np.load(os.path.join(H.CACHE, "meta.npz"))
gt2 = m["gtIdx2"].astype(np.int64) if "gtIdx2" in m.files else np.zeros(d.n, np.int64)
sv = np.load(os.path.join(H.RES, "hw", f"{name}{suf}_val.npy")).astype(np.float64)
st = np.load(os.path.join(H.RES, "hw", f"{name}{suf}_test.npy")).astype(np.float64)


def ship_max(idx, scores):
    """max score per (image, ship) over events assigned to that ship (first or second assignment)"""
    img, g1, g2, lab = d.img[idx], d.gt[idx], gt2[idx], d.labels[idx]
    k1 = (img * 256 + g1)[lab & (g1 > 0)]; s1 = scores[lab & (g1 > 0)]
    k2 = (img * 256 + g2)[g2 > 0]; s2 = scores[g2 > 0]
    k = np.concatenate([k1, k2]); s = np.concatenate([s1, s2])
    o = np.argsort(k, kind="stable"); k, s = k[o], s[o]
    u, st_ = np.unique(k, return_index=True)
    return u, np.maximum.reduceat(s, st_)


def metrics(idx, scores, thr, n_img):
    u, sm = ship_max(idx, scores)
    acc = scores >= thr
    fp = int((acc & ~d.labels[idx]).sum())
    return (sm >= thr).mean(), fp / n_img, acc.sum() / n_img, len(u)


timgs = d.split_imgs["test"]; tot = int(d.nships[timgs].sum())
ti = d.idx["test"]; vi = d.idx["val"]
u, _ = ship_max(ti, st)
reach = len(u) / tot
print(f"{name}{suf}: test {len(timgs)} images, {tot} ships, prescreen delivers an event for {len(u)} = {100*reach:.2f} %, {d.cand_per_img['test']:.0f} events/img")
print(f"{'target ret':>10} | {'ret(test)':>9} {'cascade Pd':>10} {'FA/img':>7} {'accepted/img':>12}")
for r in (0.80, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99):
    uv, smv = ship_max(vi, sv)
    thr = float(np.sort(smv)[int(np.floor((1 - r) * len(smv)))])
    ret, fa, acc, _ = metrics(ti, st, thr, len(timgs))
    print(f"{r:10.2f} | {100*ret:8.1f}% {100*ret*reach:9.1f}% {fa:7.2f} {acc:12.1f}")
