# -*- coding: utf-8 -*-
"""Cascade discriminator for the pooled-prescreen candidate set (dataset pooldetA), with
   step 1  scene context:  a second tower on a 4x4-pooled 32x32 patch (128x128 original pixels) + scalar side features
                           (event contrast, local clutter mean/s.d., image-level mean/s.d./bright/dark fractions, log event count)
   step 3  ship-level training: multiple-instance selection (only the top-K events of every ship, ranked by the CURRENT model, are positives; the other
                           events of that ship are ignored, not negatives) + periodic hard-negative mining
usage:  HWDATA=pooldetA python train_ctx.py --name pc_ctx_s1 --ctx 1 --side 1 --mil 1 --hard 1
Scores of the best epoch (by validation FA/img at 97 % ship retention) are saved to Results/hw/<name>_{val,test}.npy aligned with hwlib's d.idx splits,
so  HWDATA=pooldetA python eval_pooldet.py <name>  reports the end-to-end cascade numbers.   Only ONE training should run at a time (RAM).
"""
import argparse
import glob
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HWDATA", "pooldetA")
import hwlib as H

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--ctx", type=int, default=1)
ap.add_argument("--side", type=int, default=1)
ap.add_argument("--mil", type=int, default=0, help="1: only the top-K events per ship are positives")
ap.add_argument("--topk", type=int, default=2)
ap.add_argument("--hard", type=int, default=0, help="1: half of the negatives per epoch come from the hard pool")
ap.add_argument("--hard-top", type=float, default=0.05)
ap.add_argument("--mine-every", type=int, default=4)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--neg-ratio", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--warm", type=int, default=2, help="epochs before MIL selection uses the model score (before: highest contrast)")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XLO, XHI, MEAN, STD = H.XLO, H.XHI, H.MEAN, H.STD

# ------------------------------------------------------------------ data
d = H.HWData()
m = np.load(os.path.join(H.CACHE, "meta.npz")); gt2 = m["gtIdx2"].astype(np.int64)
fine = d.patches                                                           # (N,32,32) uint8, mmap
if os.environ.get("CTXDIR"):                                              # hardware-exact candidate set written by fixedpoint/extract_fx_events.py
    CTX = np.load(os.path.join(os.environ["CTXDIR"], "ctx.npy"), mmap_mode="r"); SIDE = np.load(os.path.join(os.environ["CTXDIR"], os.environ.get("SIDEFILE", "side.npy"))).astype(np.float32)
else:
    parts = sorted(glob.glob(os.path.join(H.RES, os.environ.get("CTXGLOB", "cnn_ctxA_p*.mat"))))
    ctx_l, side_l = [], []
    for fn in parts:
        with h5py.File(fn, "r") as f:
            ctx_l.append(np.array(f["ctx"])); side_l.append(np.array(f["side"]).T)
    CTX = np.concatenate(ctx_l, 0); SIDE = np.concatenate(side_l, 0).astype(np.float32)
assert len(CTX) == d.n == len(SIDE), (len(CTX), d.n)
tr, va, te = d.idx["train"], d.idx["val"], d.idx["test"]
mu, sd = SIDE[tr].mean(0), SIDE[tr].std(0) + 1e-6
SIDEN = ((SIDE - mu) / sd).astype(np.float32)
lab = d.labels
print(f"[{a.name}] events {d.n}: train {len(tr)} val {len(va)} test {len(te)}; ctx={a.ctx} side={a.side} mil={a.mil} hard={a.hard}", flush=True)


def batch(idx, train=False):
    idx = np.sort(idx) if not train else idx
    x = torch.from_numpy(np.ascontiguousarray(fine[idx])).to(dev).float()
    c = torch.from_numpy(np.ascontiguousarray(CTX[idx])).to(dev).float()
    s = torch.from_numpy(SIDEN[idx]).to(dev)
    x = (x * ((XHI - XLO) / 255.0) + XLO - MEAN) / STD
    c = (c * ((XHI - XLO) / 255.0) + XLO - MEAN) / STD
    if train:                                                              # dihedral augmentation, identical for both towers
        B = x.shape[0]; mk = torch.rand(B, 3, device=dev) < 0.5
        for t in (0, 1):
            v = (x, c)[t]
            v = torch.where(mk[:, 0].view(B, 1, 1), v.flip(2), v); v = torch.where(mk[:, 1].view(B, 1, 1), v.flip(1), v)
            v = torch.where(mk[:, 2].view(B, 1, 1), v.transpose(1, 2), v)
            if t == 0: x = v
            else: c = v
    return x.unsqueeze(1), c.unsqueeze(1), s, idx


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        cb = lambda i, o, k: nn.Sequential(nn.Conv2d(i, o, k), nn.BatchNorm2d(o), nn.ReLU())
        s.fine = nn.Sequential(cb(1, 16, 5), nn.MaxPool2d(2), cb(16, 32, 3), cb(32, 32, 3), nn.MaxPool2d(2), nn.Flatten())          # 32x5x5 = 800
        s.cx = nn.Sequential(cb(1, 8, 5), nn.MaxPool2d(2), cb(8, 16, 3), nn.MaxPool2d(2), cb(16, 16, 3), nn.MaxPool2d(2), nn.Flatten())   # 16x2x2 = 64
        s.sd = nn.Sequential(nn.Linear(9, 16), nn.ReLU())
        n = 800 + (64 if a.ctx else 0) + (16 if a.side else 0)
        s.head = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(s, x, c, sdv):
        z = [s.fine(x)]
        if a.ctx: z.append(s.cx(c))
        if a.side: z.append(s.sd(sdv))
        return s.head(torch.cat(z, 1)).squeeze(1)


net = Net().to(dev)
print(f"[{a.name}] params {sum(p.numel() for p in net.parameters())}", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * 1000, pct_start=0.1) if False else None


@torch.no_grad()
def score(idx, bs=4096):
    net.eval(); out = np.zeros(len(idx), np.float32)
    o = np.argsort(idx); sidx = idx[o]
    for i in range(0, len(sidx), bs):
        x, c, s, _ = batch(sidx[i:i + bs])
        out[o[i:i + bs]] = net(x, c, s).float().cpu().numpy()
    return out


def keyed_pairs(idx):
    """(event, ship-key) pairs for the positive events in idx (first and second ship assignment)"""
    e = idx[lab[idx] & (d.gt[idx] > 0)]; k1 = d.img[e] * 256 + d.gt[e]
    e2 = idx[gt2[idx] > 0]; k2 = d.img[e2] * 256 + gt2[e2]
    return np.concatenate([e, e2]), np.concatenate([k1, k2])


def ship_metrics(idx, sc, nimg):
    """val/test: per-ship max score (events of that ship), retention -> FA/img at the threshold giving the target retention (self-calibrated)"""
    e, k = keyed_pairs(idx); pos = {int(x): i for i, x in enumerate(idx)}
    s = sc[[pos[int(x)] for x in e]]
    o = np.argsort(k, kind="stable"); k, s = k[o], s[o]
    u, f = np.unique(k, return_index=True); sm = np.maximum.reduceat(s, f)
    neg = sc[~lab[idx]]
    out = {}
    for r in (0.95, 0.97, 0.98):
        thr = np.sort(sm)[int(np.floor((1 - r) * len(sm)))]
        out[r] = (neg >= thr).sum() / nimg
    return out


pe_tr, pk_tr = keyed_pairs(tr)
neg_tr = tr[~lab[tr]]
n_val_img, n_test_img = len(d.split_imgs["val"]), len(d.split_imgs["test"])
best = (1e9, -1); hard_pool = None
t0 = time.time()
for ep in range(a.epochs):
    # ---- positive set
    if a.mil:
        sc_p = score(pe_tr) if ep >= a.warm else SIDE[pe_tr, 0]          # before the warm-up the highest-contrast events are taken
        o = np.lexsort((-sc_p, pk_tr)); ks = pk_tr[o]; first = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
        rank = np.arange(len(ks)) - np.repeat(first, np.diff(np.r_[first, len(ks)]))
        pos = np.unique(pe_tr[o][rank < a.topk])
    else:
        pos = np.unique(pe_tr)
    # ---- negatives
    n_neg = a.neg_ratio * len(pos)
    if a.hard and ep % a.mine_every == 0 and ep >= a.warm:
        sn = score(neg_tr); hard_pool = neg_tr[np.argsort(-sn)[:int(a.hard_top * len(neg_tr))]]
    if a.hard and hard_pool is not None:
        nh = n_neg // 2
        neg = np.concatenate([np.random.choice(hard_pool, nh, replace=True), np.random.choice(neg_tr, min(n_neg - nh, len(neg_tr)), replace=False)])
    else:
        neg = np.random.choice(neg_tr, min(n_neg, len(neg_tr)), replace=False)
    ids = np.concatenate([pos, neg]); np.random.shuffle(ids)
    net.train(); tot = 0.0; nb = 0
    for g in opt.param_groups: g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * ep / a.epochs)))
    for i in range(0, len(ids), a.bs):
        b = ids[i:i + a.bs]; x, c, s, bi = batch(b, train=True)
        y = torch.from_numpy(lab[b].astype(np.float32)).to(dev)
        loss = F.binary_cross_entropy_with_logits(net(x, c, s), y)
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
    sv = score(va); mv = ship_metrics(va, sv, n_val_img)
    flag = ""
    if mv[0.97] < best[0]:
        best = (mv[0.97], ep); flag = " *"
        torch.save(net.state_dict(), os.path.join(H.RES, "hw", f"{a.name}.pt"))
        np.save(os.path.join(H.RES, "hw", f"{a.name}_val.npy"), sv)
    print(f"[{a.name}] ep {ep:2d} loss {tot/nb:.4f} pos {len(pos)} | val FA/img @ret 95/97/98: {mv[0.95]:.2f} {mv[0.97]:.2f} {mv[0.98]:.2f} ({time.time()-t0:.0f}s){flag}", flush=True)
net.load_state_dict(torch.load(os.path.join(H.RES, "hw", f"{a.name}.pt")))
np.save(os.path.join(H.RES, "hw", f"{a.name}_test.npy"), score(te))
mt = ship_metrics(te, np.load(os.path.join(H.RES, "hw", f"{a.name}_test.npy")), n_test_img)
np.save(os.path.join(H.RES, "hw", f"{a.name}_all.npy"), score(np.arange(d.n)))        # scores of EVERY event (train events are in-sample)
print(f"[{a.name}] DONE best_ep={best[1]} VAL FA@97={best[0]:.2f} | TEST FA/img @ret 95/97/98 (test-calibrated): {mt[0.95]:.2f} {mt[0.97]:.2f} {mt[0.98]:.2f}", flush=True)
