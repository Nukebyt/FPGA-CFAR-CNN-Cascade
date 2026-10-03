# -*- coding: utf-8 -*-
"""INT8 version of the context cascade discriminator (train_ctx.py --ctx 1 --side 1) with ship-level (MIL) loss and hard negatives.
Network (hardware program of cnn_core_ctx.v):
   fine tower  : conv5 1->16 +pool, conv3 16->32, conv3 32->32 +pool           -> 32x5x5 = 800
   context tower: conv5 1->8 +pool, conv3 8->16 +pool, conv3 16->16 +pool       -> 16x2x2 = 64     (input: 32x32 patch of P4 = 2x2 pool of the pooled store)
   side         : fc 9 -> 16 + ReLU                                                                      (inputs: hardware-exact uint8 codes, side_fx.py)
   head         : fc (800 + 64 + 16) -> 64 + ReLU, fc 64 -> 1  (the three feature segments share ONE activation scale, so the three partial sums add as integers)
Integer arithmetic as in quant_hw.py: per-channel int8 weights, int32 accumulation, per-channel 15-bit multiplier + shift requant, uint8 activations.
usage: HWDATA=pooldetfxA CTXDIR=<hw_cache_pooldetfxA> SIDEFILE=side_codes.npy python train_q3.py --src pf_full_s2 --name pf_full_q8 --epochs 8
Writes Results/hw/<name>_{val,test,all}.npy (integer logits x scale) and <name>_int3.pt.   ONE training at a time (RAM)."""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HWDATA", "pooldetfxA")
import hwlib as H

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
M_BITS = 15
CDIR = os.environ["CTXDIR"]

d = H.HWData()
m = np.load(os.path.join(H.CACHE, "meta.npz")); gt2 = m["gtIdx2"].astype(np.int64)
fine = d.patches
CTX = np.load(os.path.join(CDIR, "ctx.npy"), mmap_mode="r")
SIDE = np.load(os.path.join(CDIR, os.environ.get("SIDEFILE", "side_codes.npy")))              # uint8 codes
assert SIDE.dtype == np.uint8 and len(SIDE) == d.n
tr, va, te = d.idx["train"], d.idx["val"], d.idx["test"]
lab = d.labels
n_val_img, n_test_img = len(d.split_imgs["val"]), len(d.split_imgs["test"])
Sf = SIDE[tr].astype(np.float32); mu, sd = Sf.mean(0), Sf.std(0) + 1e-6                  # train_ctx's z-score (same data, same stats)

# ------------------------------------------------------------------ float checkpoint -> folded layers
cb = lambda i, o, k: nn.Sequential(nn.Conv2d(i, o, k), nn.BatchNorm2d(o), nn.ReLU())
class FNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.fine = nn.Sequential(cb(1, 16, 5), nn.MaxPool2d(2), cb(16, 32, 3), cb(32, 32, 3), nn.MaxPool2d(2), nn.Flatten())
        s.cx = nn.Sequential(cb(1, 8, 5), nn.MaxPool2d(2), cb(8, 16, 3), nn.MaxPool2d(2), cb(16, 16, 3), nn.MaxPool2d(2), nn.Flatten())
        s.sd = nn.Sequential(nn.Linear(9, 16), nn.ReLU())
        s.head = nn.Sequential(nn.Linear(880, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
fn = FNet(); fn.load_state_dict(torch.load(os.path.join(H.RES, "hw", a.src + ".pt"), map_location="cpu")); fn.eval()


def fold_tower(seq):
    out = []; f = list(seq); i = 0
    while i < len(f):
        blk = f[i]
        if isinstance(blk, nn.Sequential):
            conv, bn = blk[0], blk[1]
            g = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
            W = conv.weight.detach() * g.view(-1, 1, 1, 1); b = (conv.bias.detach() - bn.running_mean) * g + bn.bias.detach()
            pool = (i + 1 < len(f)) and isinstance(f[i + 1], nn.MaxPool2d)
            out.append(dict(kind="conv", W=W.double().numpy(), b=b.double().numpy(), pool=pool, relu=True))
        i += 1
    A = (XHI - XLO) / 255.0 / STD; Cc = (XLO - MEAN) / STD                      # input affine x = A*q + Cc folded into layer 1
    out[0]["b"] = out[0]["b"] + Cc * out[0]["W"].sum(axis=(1, 2, 3)); out[0]["W"] = out[0]["W"] * A
    return out


FL, CL = fold_tower(fn.fine), fold_tower(fn.cx)
Ws, bs = fn.sd[0].weight.detach().double().numpy(), fn.sd[0].bias.detach().double().numpy()          # side z = (code - mu)/sd folded into the Linear
SL = dict(kind="fc", W=Ws / sd.astype(np.float64)[None, :], b=bs - (Ws * (mu / sd).astype(np.float64)[None, :]).sum(1), pool=False, relu=True)
hl = [mm for mm in fn.head if isinstance(mm, nn.Linear)]
H1 = dict(kind="fc", W=hl[0].weight.detach().double().numpy(), b=hl[0].bias.detach().double().numpy(), pool=False, relu=True)
H2 = dict(kind="fc", W=hl[1].weight.detach().double().numpy(), b=hl[1].bias.detach().double().numpy(), pool=False, relu=False)


def rnd(z):
    return z + (torch.round(z) - z).detach()


class FQ3(nn.Module):
    """differentiable model of the integer pipeline.  log-scale parameters: f0 f1 (fine), c0 c1 (ctx), cat (fine out, ctx out, side out, all tied), h1."""
    def __init__(s, cal):
        super().__init__()
        s.W = nn.ParameterDict(); s.b = nn.ParameterDict()
        def reg(k, L):
            s.W[k] = nn.Parameter(torch.tensor(L["W"], dtype=torch.float32)); s.b[k] = nn.Parameter(torch.tensor(L["b"], dtype=torch.float32))
        for i, L in enumerate(FL): reg(f"f{i}", L)
        for i, L in enumerate(CL): reg(f"c{i}", L)
        reg("s", SL); reg("h1", H1); reg("h2", H2)
        s.ls = nn.ParameterDict({k: nn.Parameter(torch.tensor(float(np.log(v)), dtype=torch.float32)) for k, v in cal.items()})

    @staticmethod
    def qw(W):
        sc = W.abs().amax(dim=tuple(range(1, W.dim()))).clamp_min(1e-8) / 127.0
        sc = sc.view(-1, *([1] * (W.dim() - 1)))
        return torch.clamp(rnd(W / sc), -127, 127) * sc

    def aq(s, x, key):
        sc = torch.exp(s.ls[key]); return torch.clamp(rnd(F.relu(x) / sc), 0, 255) * sc

    def tower(s, x, pre, layers):
        x = x.unsqueeze(1)
        for i, L in enumerate(layers):
            x = F.conv2d(x, s.qw(s.W[f"{pre}{i}"]), s.b[f"{pre}{i}"])
            x = s.aq(x, "cat" if i == len(layers) - 1 else f"{pre}{i}")
            if L["pool"]: x = F.max_pool2d(x, 2)
        return x.flatten(1)

    def forward(s, xf, xc, xs):
        f = s.tower(xf, "f", FL); c = s.tower(xc, "c", CL)
        sv = s.aq(F.linear(xs, s.qw(s.W["s"]), s.b["s"]), "cat")
        h = s.aq(F.linear(torch.cat([f, c, sv], 1), s.qw(s.W["h1"]), s.b["h1"]), "h1")
        return F.linear(h, s.qw(s.W["h2"]), s.b["h2"]).squeeze(1)


@torch.no_grad()
def calibrate():
    idx = np.sort(tr[::53][:20000])
    xf = torch.from_numpy(np.ascontiguousarray(fine[idx])).float(); xc = torch.from_numpy(np.ascontiguousarray(CTX[idx])).float(); xs = torch.from_numpy(SIDE[idx]).float()
    def run(x, layers):
        x = x.unsqueeze(1); sc = []
        for L in layers:
            x = F.relu(F.conv2d(x, torch.tensor(L["W"], dtype=torch.float32), torch.tensor(L["b"], dtype=torch.float32))); sc.append(float(np.percentile(x.numpy(), 99.9)) / 255.0 + 1e-9)
            if L["pool"]: x = F.max_pool2d(x, 2)
        return x.flatten(1), sc
    f, sf = run(xf, FL); c, sc_ = run(xc, CL)
    sv = F.relu(F.linear(xs, torch.tensor(SL["W"], dtype=torch.float32), torch.tensor(SL["b"], dtype=torch.float32)))
    cat = torch.cat([f, c, sv], 1)
    h = F.relu(F.linear(cat, torch.tensor(H1["W"], dtype=torch.float32), torch.tensor(H1["b"], dtype=torch.float32)))
    cal = dict(f0=sf[0], f1=sf[1], c0=sc_[0], c1=sc_[1], cat=max(sf[2], sc_[2], float(np.percentile(sv.numpy(), 99.9)) / 255.0 + 1e-9), h1=float(np.percentile(h.numpy(), 99.9)) / 255.0 + 1e-9)
    return cal


fq = FQ3(calibrate()).to(dev)


def batch(idx, train=False):
    idx = np.sort(idx) if not train else idx
    xf = torch.from_numpy(np.ascontiguousarray(fine[idx])).to(dev).float(); xc = torch.from_numpy(np.ascontiguousarray(CTX[idx])).to(dev).float()
    xs = torch.from_numpy(SIDE[idx]).to(dev).float()
    if train:                                                                   # dihedral augmentation, identical for both towers
        B = xf.shape[0]; mk = torch.rand(B, 3, device=dev) < 0.5
        def aug(v):
            v = torch.where(mk[:, 0].view(B, 1, 1), v.flip(2), v); v = torch.where(mk[:, 1].view(B, 1, 1), v.flip(1), v)
            return torch.where(mk[:, 2].view(B, 1, 1), v.transpose(1, 2), v)
        xf, xc = aug(xf), aug(xc)
    return xf, xc, xs, idx


@torch.no_grad()
def score_fq(idx, bs=4096):
    fq.eval(); out = np.zeros(len(idx), np.float32); o = np.argsort(idx); s = idx[o]
    for i in range(0, len(s), bs):
        xf, xc, xs, _ = batch(s[i:i + bs]); out[o[i:i + bs]] = fq(xf, xc, xs).float().cpu().numpy()
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
        b = ids[i:i + a.bs]; xf, xc, xs, _ = batch(b, train=True)
        y = torch.from_numpy(lab[b].astype(np.float32)).to(dev)
        loss = F.binary_cross_entropy_with_logits(fq(xf, xc, xs), y)
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
    mv = ship_metrics(va, score_fq(va), n_val_img); flag = ""
    if mv[0.97] < best[0]:
        best = (mv[0.97], ep); flag = " *"; torch.save(fq.state_dict(), os.path.join(H.RES, "hw", a.name + "_fq.pt"))
    print(f"[{a.name}] QAT ep {ep:2d} loss {tot/nb:.4f} | val FA/img @ret 95/97/98: {mv[0.95]:.2f} {mv[0.97]:.2f} {mv[0.98]:.2f} ({time.time()-t0:.0f}s){flag}", flush=True)
fq.load_state_dict(torch.load(os.path.join(H.RES, "hw", a.name + "_fq.pt")))


# ------------------------------------------------------------------ integer export
def exp_layer(key, s_in, relu, pool, kind, s_out_key=None):
    W = fq.W[key].detach().cpu().double().numpy(); b = fq.b[key].detach().cpu().double().numpy()
    sw = np.maximum(np.abs(W).reshape(W.shape[0], -1).max(axis=1), 1e-12) / 127.0
    Wi = np.clip(np.round(W / sw.reshape(-1, *([1] * (W.ndim - 1)))), -127, 127).astype(np.int64)
    bi = np.round(b / (s_in * sw)).astype(np.int64)
    dct = dict(kind=kind, pool=pool, relu=relu, W=Wi, b=bi, sw=sw, s_in=s_in)
    if relu:
        s_out = float(np.exp(fq.ls[s_out_key].detach().cpu().item())); ratio = s_in * sw / s_out
        shift = int(np.floor(np.log2((2 ** M_BITS - 1) / ratio.max())))
        dct.update(M=np.round(ratio * 2.0 ** shift).astype(np.int64), S=shift, s_out=s_out)
    return dct


def export():
    s_cat = float(np.exp(fq.ls["cat"].detach().cpu().item()))
    out = dict(fine=[], ctx=[], side=None, h1=None, h2=None)
    s_in = 1.0
    for i, L in enumerate(FL):
        e = exp_layer(f"f{i}", s_in, True, L["pool"], "conv", "cat" if i == len(FL) - 1 else f"f{i}"); out["fine"].append(e); s_in = e["s_out"]
    s_in = 1.0
    for i, L in enumerate(CL):
        e = exp_layer(f"c{i}", s_in, True, L["pool"], "conv", "cat" if i == len(CL) - 1 else f"c{i}"); out["ctx"].append(e); s_in = e["s_out"]
    out["side"] = exp_layer("s", 1.0, True, False, "fc", "cat")
    out["h1"] = exp_layer("h1", s_cat, True, False, "fc", "h1")
    out["h2"] = exp_layer("h2", out["h1"]["s_out"], False, False, "fc")
    return out


ints = export()
torch.save({"ints3": ints}, os.path.join(H.RES, "hw", a.name + "_int3.pt"))
SC = float(ints["h2"]["s_in"] * ints["h2"]["sw"][0])


@torch.no_grad()
def int_scores(idx, bs=2048):
    """bit-exact model of the RTL datapath (float32 convolutions are exact below 2^24; FC and requant in float64 / int64)"""
    out = np.zeros(len(idx), np.float64); o = np.argsort(idx); s = idx[o]
    def req(acc, L):
        M = torch.from_numpy(L["M"]).to(dev).view(1, -1, *([1] * (acc.dim() - 2)))
        return torch.clamp((acc * M + (1 << (L["S"] - 1))) >> L["S"], 0, 255)
    def tower(x, layers):
        x = x.long().unsqueeze(1)
        for L in layers:
            Wt = torch.from_numpy(L["W"]).to(dev); bt = torch.from_numpy(L["b"]).to(dev)
            x = req(F.conv2d(x.float(), Wt.float()).round().long() + bt.view(1, -1, 1, 1), L)
            if L["pool"]: x = F.max_pool2d(x.float(), 2).long()
        return x.flatten(1)
    def fc(x, L):
        return (x.double() @ torch.from_numpy(L["W"]).to(dev).double().T).round().long() + torch.from_numpy(L["b"]).to(dev)
    for i in range(0, len(s), bs):
        xf, xc, xs, _ = batch(s[i:i + bs])
        f = tower(xf, ints["fine"]); c = tower(xc, ints["ctx"])
        sv = req(fc(xs.long(), ints["side"]), ints["side"])
        h = req(fc(torch.cat([f, c, sv], 1), ints["h1"]), ints["h1"])
        out[o[i:i + bs]] = fc(h, ints["h2"]).reshape(-1).double().cpu().numpy()
    return out * SC


sv_, st_ = int_scores(va), int_scores(te)
np.save(os.path.join(H.RES, "hw", a.name + "_val.npy"), sv_); np.save(os.path.join(H.RES, "hw", a.name + "_test.npy"), st_)
np.save(os.path.join(H.RES, "hw", a.name + "_all.npy"), int_scores(np.arange(d.n)))
mt = ship_metrics(te, st_, n_test_img); mvv = ship_metrics(va, sv_, n_val_img)
print(f"[{a.name}] INT8 DONE  VAL FA/img @ret 95/97/98: {mvv[0.95]:.2f} {mvv[0.97]:.2f} {mvv[0.98]:.2f} | TEST (test-calibrated): {mt[0.95]:.2f} {mt[0.97]:.2f} {mt[0.98]:.2f}", flush=True)
