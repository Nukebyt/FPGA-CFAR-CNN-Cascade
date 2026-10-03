# -*- coding: utf-8 -*-
"""INT8 version of the single-tower cascade discriminator trained with train_ctx.py (--ctx 0 --side 0): same ship-level (MIL) loss and hard-negative mining,
now through the integer datapath of the RTL (per-channel int8 weights, uint8 activations with learned scales, 15-bit requant multiplier).
  1. load the float checkpoint Results/hw/<src>.pt (state_dict of train_ctx.Net), fold BN + input affine into the layers   (quant_hw.fold-equivalent)
  2. QAT fine-tune with the same MIL + hard-negative sampling   (quant_hw.FQNet = differentiable model of the integer pipeline)
  3. integer-exact scoring of every event on the GPU (float32 convolutions are exact for these ranges, requant in int64), checked against quant_hw.int_forward
  4. writes Results/hw/<name>_{val,test,all}.npy (integer logits x logit scale) + <name>_int.pt ({'ints': ...}) for gen_cnn_rtl_q4.py / transpose_export.py
usage: HWDATA=pooldetfxA python train_q.py --src pf_plain_s1 --name pf_plain_s1_q8 --epochs 8
ONE training at a time (RAM)."""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HWDATA", "pooldetfxA")
import hwlib as H
import quant_hw as Q

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=512); ap.add_argument("--topk", type=int, default=2)
ap.add_argument("--neg-ratio", type=int, default=16); ap.add_argument("--hard-top", type=float, default=0.05); ap.add_argument("--mine-every", type=int, default=2)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
XLO, XHI, MEAN, STD = H.XLO, H.XHI, H.MEAN, H.STD

# ------------------------------------------------------------------ data
d = H.HWData()
m = np.load(os.path.join(H.CACHE, "meta.npz")); gt2 = m["gtIdx2"].astype(np.int64)
fine = d.patches
SIDE0 = np.load(os.path.join(H.CACHE, "side.npy"))[:, 0]                       # event contrast (ln units) for the MIL warm-up
tr, va, te = d.idx["train"], d.idx["val"], d.idx["test"]
lab = d.labels
n_val_img, n_test_img = len(d.split_imgs["val"]), len(d.split_imgs["test"])

# ------------------------------------------------------------------ float checkpoint -> folded layers
sd = torch.load(os.path.join(H.RES, "hw", a.src + ".pt"), map_location="cpu")
cb = lambda i, o, k: nn.Sequential(nn.Conv2d(i, o, k), nn.BatchNorm2d(o), nn.ReLU())
class FNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.fine = nn.Sequential(cb(1, 16, 5), nn.MaxPool2d(2), cb(16, 32, 3), cb(32, 32, 3), nn.MaxPool2d(2), nn.Flatten())
        s.cx = nn.Sequential(cb(1, 8, 5), nn.MaxPool2d(2), cb(8, 16, 3), nn.MaxPool2d(2), cb(16, 16, 3), nn.MaxPool2d(2), nn.Flatten())
        s.sd = nn.Sequential(nn.Linear(9, 16), nn.ReLU())
        s.head = nn.Sequential(nn.Linear(800, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
fn = FNet()
fn.load_state_dict(sd)
assert tuple(fn.head[0].weight.shape) == (64, 800), "this trainer is for the single-tower model (train_ctx.py --ctx 0 --side 0)"
fn.eval()


def fold_float():
    layers = []
    f = list(fn.fine)
    i = 0
    while i < len(f):
        blk = f[i]
        if isinstance(blk, nn.Sequential):
            conv, bn = blk[0], blk[1]
            g = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
            W = conv.weight.detach() * g.view(-1, 1, 1, 1)
            b = (conv.bias.detach() - bn.running_mean) * g + bn.bias.detach()
            pool = (i + 1 < len(f)) and isinstance(f[i + 1], nn.MaxPool2d)
            layers.append(dict(kind="conv", W=W.double().numpy(), b=b.double().numpy(), pool=pool, relu=True))
        i += 1
    h = [mm for mm in fn.head if isinstance(mm, nn.Linear)]
    layers.append(dict(kind="fc", W=h[0].weight.detach().double().numpy(), b=h[0].bias.detach().double().numpy(), pool=False, relu=True))
    layers.append(dict(kind="fc", W=h[1].weight.detach().double().numpy(), b=h[1].bias.detach().double().numpy(), pool=False, relu=False))
    A = (XHI - XLO) / 255.0 / STD; Cc = (XLO - MEAN) / STD                      # x_norm = A*q + Cc  (folded into layer 1)
    L0 = layers[0]
    L0["b"] = L0["b"] + Cc * L0["W"].sum(axis=(1, 2, 3)); L0["W"] = L0["W"] * A
    return layers


layers = fold_float()
# sanity: folded float net on raw codes == original float net
j = va[:2048]
with torch.no_grad():
    q = torch.from_numpy(np.ascontiguousarray(fine[np.sort(j)])).float()
    x0 = ((q * ((XHI - XLO) / 255.0) + XLO - MEAN) / STD).unsqueeze(1)
    ref = fn.head(fn.fine(x0)).squeeze(1).numpy()
    x = q.unsqueeze(1)
    for l in layers:
        W = torch.tensor(l["W"], dtype=torch.float32); b = torch.tensor(l["b"], dtype=torch.float32)
        x = F.conv2d(x, W, b) if l["kind"] == "conv" else F.linear(x.flatten(1), W, b)
        if l["relu"]: x = F.relu(x)
        if l["pool"]: x = F.max_pool2d(x, 2)
    print(f"fold check: max |folded - original| = {np.abs(x.squeeze(1).numpy() - ref).max():.2e}", flush=True)

cal = fine[np.sort(tr[::53][:20000])].astype(np.int64)
fq = Q.FQNet(layers, Q.calibrate(layers, cal)).to(dev)


# ------------------------------------------------------------------ helpers
def codes(idx, train=False):
    idx = np.sort(idx) if not train else idx
    x = torch.from_numpy(np.ascontiguousarray(fine[idx])).to(dev).float()
    if train:
        B = x.shape[0]; mk = torch.rand(B, 3, device=dev) < 0.5
        x = torch.where(mk[:, 0].view(B, 1, 1), x.flip(2), x); x = torch.where(mk[:, 1].view(B, 1, 1), x.flip(1), x)
        x = torch.where(mk[:, 2].view(B, 1, 1), x.transpose(1, 2), x)
    return x, idx


@torch.no_grad()
def score_fq(idx, bs=4096):
    fq.eval(); out = np.zeros(len(idx), np.float32); o = np.argsort(idx); s = idx[o]
    for i in range(0, len(s), bs):
        x, _ = codes(s[i:i + bs]); out[o[i:i + bs]] = fq(x).float().cpu().numpy()
    return out


def keyed_pairs(idx):
    e = idx[lab[idx] & (d.gt[idx] > 0)]; k1 = d.img[e] * 256 + d.gt[e]
    e2 = idx[gt2[idx] > 0]; k2 = d.img[e2] * 256 + gt2[e2]
    return np.concatenate([e, e2]), np.concatenate([k1, k2])


def ship_metrics(idx, sc, nimg):
    e, k = keyed_pairs(idx); pos = {int(x): i for i, x in enumerate(idx)}
    s = sc[[pos[int(x)] for x in e]]
    o = np.argsort(k, kind="stable"); k, s = k[o], s[o]
    u, f = np.unique(k, return_index=True); sm = np.maximum.reduceat(s, f)
    neg = sc[~lab[idx]]; out = {}
    for r in (0.95, 0.97, 0.98):
        thr = np.sort(sm)[int(np.floor((1 - r) * len(sm)))]
        out[r] = (neg >= thr).sum() / nimg
    return out


pe_tr, pk_tr = keyed_pairs(tr); neg_tr = tr[~lab[tr]]
m0 = ship_metrics(va, score_fq(va), n_val_img)
print(f"PTQ (before QAT) val FA/img @ret 95/97/98: {m0[0.95]:.2f} {m0[0.97]:.2f} {m0[0.98]:.2f}", flush=True)

# ------------------------------------------------------------------ QAT with MIL + hard negatives
opt = torch.optim.AdamW(fq.parameters(), lr=a.lr, weight_decay=0.0)
best = (m0[0.97], -1); torch.save(fq.state_dict(), os.path.join(H.RES, "hw", a.name + "_fq.pt")); hard_pool = None; t0 = time.time()
for ep in range(a.epochs):
    sc_p = score_fq(pe_tr)
    o = np.lexsort((-sc_p, pk_tr)); ks = pk_tr[o]; first = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    rank = np.arange(len(ks)) - np.repeat(first, np.diff(np.r_[first, len(ks)]))
    pos = np.unique(pe_tr[o][rank < a.topk])
    n_neg = a.neg_ratio * len(pos)
    if ep % a.mine_every == 0:
        sn = score_fq(neg_tr); hard_pool = neg_tr[np.argsort(-sn)[:int(a.hard_top * len(neg_tr))]]
    nh = n_neg // 2
    neg = np.concatenate([np.random.choice(hard_pool, nh, replace=True), np.random.choice(neg_tr, min(n_neg - nh, len(neg_tr)), replace=False)])
    ids = np.concatenate([pos, neg]); np.random.shuffle(ids)
    fq.train(); tot = 0.0; nb = 0
    for g in opt.param_groups: g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * ep / a.epochs)))
    for i in range(0, len(ids), a.bs):
        b = ids[i:i + a.bs]; x, bi = codes(b, train=True)
        y = torch.from_numpy(lab[b].astype(np.float32)).to(dev)
        loss = F.binary_cross_entropy_with_logits(fq(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
    mv = ship_metrics(va, score_fq(va), n_val_img); flag = ""
    if mv[0.97] < best[0]:
        best = (mv[0.97], ep); flag = " *"; torch.save(fq.state_dict(), os.path.join(H.RES, "hw", a.name + "_fq.pt"))
    print(f"[{a.name}] QAT ep {ep:2d} loss {tot/nb:.4f} | val FA/img @ret 95/97/98: {mv[0.95]:.2f} {mv[0.97]:.2f} {mv[0.98]:.2f} ({time.time()-t0:.0f}s){flag}", flush=True)
fq.load_state_dict(torch.load(os.path.join(H.RES, "hw", a.name + "_fq.pt")))

# ------------------------------------------------------------------ integer-exact scoring (GPU) and export
ints = Q.export_int(fq)
torch.save({"ints": ints}, os.path.join(H.RES, "hw", a.name + "_int.pt"))


@torch.no_grad()
def int_scores(idx, bs=2048):
    """bit-exact model of the RTL datapath: conv accumulators are exact in float32 (<= 9.4e6 < 2^24); bias, requant and the FC layers in int64"""
    out = np.zeros(len(idx), np.float64); o = np.argsort(idx); s = idx[o]
    for i in range(0, len(s), bs):
        x = torch.from_numpy(np.ascontiguousarray(fine[s[i:i + bs]])).to(dev).long().unsqueeze(1)
        for L in ints:
            Wt = torch.from_numpy(L["W"]).to(dev); bt = torch.from_numpy(L["b"]).to(dev)
            if L["kind"] == "conv":
                acc = F.conv2d(x.float(), Wt.float()).round().long() + bt.view(1, -1, 1, 1)
            else:
                acc = (x.flatten(1).double() @ Wt.double().T).round().long() + bt
            if L["relu"]:
                M = torch.from_numpy(L["M"]).to(dev).view(1, -1, *([1] * (acc.dim() - 2)))
                x = torch.clamp((acc * M + (1 << (L["S"] - 1))) >> L["S"], 0, 255)
            else:
                x = acc
            if L["pool"]:
                x = F.max_pool2d(x.float(), 2).long()
        out[o[i:i + bs]] = x.reshape(-1).double().cpu().numpy()
    return out * float(ints[-1]["s_in"] * ints[-1]["sw"][0])


chk = np.sort(va[:300])
ref = Q.int_forward(ints, np.ascontiguousarray(fine[chk]).astype(np.int64)); mine = int_scores(chk)
print("GPU integer model vs quant_hw.int_forward on 300 events: max |diff| =", float(np.abs(ref - mine).max()), flush=True)
sv, st = int_scores(va), int_scores(te)
np.save(os.path.join(H.RES, "hw", a.name + "_val.npy"), sv); np.save(os.path.join(H.RES, "hw", a.name + "_test.npy"), st)
np.save(os.path.join(H.RES, "hw", a.name + "_all.npy"), int_scores(np.arange(d.n)))
mt = ship_metrics(te, st, n_test_img); mvv = ship_metrics(va, sv, n_val_img)
print(f"[{a.name}] INT8 DONE  VAL FA/img @ret 95/97/98: {mvv[0.95]:.2f} {mvv[0.97]:.2f} {mvv[0.98]:.2f} | TEST (test-calibrated): {mt[0.95]:.2f} {mt[0.97]:.2f} {mt[0.98]:.2f}", flush=True)
