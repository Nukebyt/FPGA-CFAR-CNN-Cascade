# -*- coding: utf-8 -*-
"""INT8 quantization-aware training, integer-exact reference model, and RTL
weight export for a trained hwlib.Net checkpoint.

Hardware arithmetic being modelled (what the RTL MAC array will do):
  * input  : the stored uint8 patch q (0..255) -- the 'global' normalization
             (x-MEAN)/STD is an affine map of q, folded into layer 1.
  * conv/fc: int8 weights (per-output-channel scale), int32 accumulate,
             int32 bias.  BatchNorm is folded into weight+bias BEFORE
             quantization (zero runtime cost).
  * requant: y = clamp((acc * M[c]) >> S, 0, 255)   (ReLU is the lower clamp),
             M[c] a 15-bit integer multiplier per output channel.
  * pool   : 2x2 max on the uint8 activations (commutes with requant).
  * output : final layer keeps its int32 accumulator; the detection decision
             is acc >= THETA (no sigmoid in hardware).

Modes
  python quant_hw.py qat    <ckpt> [--epochs N]   fake-quant fine-tune, save <ckpt>_qat.pt
  python quant_hw.py eval   <ckpt>                PTQ + integer-exact metrics on val/test
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hwlib as H

M_BITS = 15
DELTA = (H.XHI - H.XLO) / 255.0


# ------------------------------------------------------------------ folding
def fold(model, cfg):
    """Float Net -> list of layer dicts with BN folded and the input affine
    folded into layer 1 (so layer 1 consumes the raw uint8 patch)."""
    assert cfg["norm"] == "global", "only 'global' normalization folds into layer 1"
    layers = []
    feats = list(model.features)
    i = 0
    while i < len(feats):
        conv, bn = feats[i], feats[i + 1]
        assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
        i += 3                                   # conv, bn, relu
        pool = i < len(feats) and isinstance(feats[i], nn.MaxPool2d)
        if pool:
            i += 1
        g = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
        W = conv.weight.detach() * g.view(-1, 1, 1, 1)
        b = bn.bias.detach() - bn.running_mean * g
        layers.append(dict(kind="conv", W=W.double().numpy(), b=b.double().numpy(), pool=pool, relu=True))
    cl = [m for m in model.classifier if isinstance(m, nn.Linear)]
    for j, lin in enumerate(cl):
        layers.append(dict(kind="fc", W=lin.weight.detach().double().numpy(),
                           b=lin.bias.detach().double().numpy(), pool=False, relu=(j < len(cl) - 1)))
    # fold input affine x_norm = a*q + c into layer 1 (valid conv: constant per channel)
    a = DELTA / H.STD
    c = (H.XLO - H.MEAN) / H.STD
    L0 = layers[0]
    L0["b"] = L0["b"] + c * L0["W"].sum(axis=(1, 2, 3))
    L0["W"] = L0["W"] * a
    return layers


# ------------------------------------------------------ fake-quant (training)
def rnd(z):
    return z + (torch.round(z) - z).detach()


class FQNet(nn.Module):
    """Differentiable simulation of the integer pipeline (per-channel int8
    weights, uint8 activations with learned scales)."""

    def __init__(self, layers, act_scales):
        super().__init__()
        self.meta = [(l["kind"], l["pool"], l["relu"]) for l in layers]
        self.W = nn.ParameterList([nn.Parameter(torch.tensor(l["W"], dtype=torch.float32)) for l in layers])
        self.b = nn.ParameterList([nn.Parameter(torch.tensor(l["b"], dtype=torch.float32)) for l in layers])
        # one log-scale per ReLU output (the last layer has none)
        self.ls = nn.ParameterList([nn.Parameter(torch.tensor(float(np.log(s)), dtype=torch.float32))
                                    for s in act_scales])

    def qw(self, W):
        s = W.abs().amax(dim=tuple(range(1, W.dim()))).clamp_min(1e-8) / 127.0
        s = s.view(-1, *([1] * (W.dim() - 1)))
        return torch.clamp(rnd(W / s), -127, 127) * s

    def forward(self, q):
        # q: (B,S,S) float holding raw 0..255 codes
        x = q.unsqueeze(1)
        k = 0
        for li, (kind, pool, relu) in enumerate(self.meta):
            W = self.qw(self.W[li])
            if kind == "conv":
                x = F.conv2d(x, W, self.b[li])
            else:
                x = F.linear(x.flatten(1), W, self.b[li])
            if relu:
                s = torch.exp(self.ls[k]); k += 1
                x = torch.clamp(rnd(F.relu(x) / s), 0, 255) * s
            if pool:
                x = F.max_pool2d(x, 2)
        return x.squeeze(1)


def calibrate(layers, xq, pct=99.9):
    """Activation scales from float forward of the folded net on raw codes."""
    x = torch.tensor(xq, dtype=torch.float32).unsqueeze(1)
    scales = []
    with torch.no_grad():
        for l in layers:
            W = torch.tensor(l["W"], dtype=torch.float32)
            b = torch.tensor(l["b"], dtype=torch.float32)
            x = F.conv2d(x, W, b) if l["kind"] == "conv" else F.linear(x.flatten(1), W, b)
            if l["relu"]:
                x = F.relu(x)
                scales.append(float(np.percentile(x.numpy(), pct)) / 255.0 + 1e-9)
            if l["pool"]:
                x = F.max_pool2d(x, 2)
    return scales


# --------------------------------------------------- integer-exact reference
def export_int(fq, S=None):
    """Fake-quant net -> integer tensors + requant multipliers."""
    out = []
    s_in = 1.0                                     # layer-1 input is the raw code
    k = 0
    for li, (kind, pool, relu) in enumerate(fq.meta):
        W = fq.W[li].detach().cpu().double().numpy()
        b = fq.b[li].detach().cpu().double().numpy()
        sw = np.maximum(np.abs(W).reshape(W.shape[0], -1).max(axis=1), 1e-12) / 127.0
        Wi = np.clip(np.round(W / sw.reshape(-1, *([1] * (W.ndim - 1)))), -127, 127).astype(np.int64)
        bi = np.round(b / (s_in * sw)).astype(np.int64)
        d = dict(kind=kind, pool=pool, relu=relu, W=Wi, b=bi, sw=sw, s_in=s_in)
        if relu:
            s_out = float(np.exp(fq.ls[k].detach().cpu().item())); k += 1
            ratio = s_in * sw / s_out              # per-channel real multiplier
            shift = int(np.floor(np.log2((2 ** M_BITS - 1) / ratio.max())))
            d["M"] = np.round(ratio * 2.0 ** shift).astype(np.int64)
            d["S"] = shift
            d["s_out"] = s_out
            s_in = s_out
        out.append(d)
    return out


def _conv_int(x, W):
    """x (B,Cin,H,W) int64, W (Cout,Cin,k,k) int64, valid conv -> (B,Cout,H',W')."""
    xt = torch.from_numpy(x)
    Wt = torch.from_numpy(W)
    B, C, Hh, Ww = xt.shape
    Co, _, k, _ = Wt.shape
    cols = F.unfold(xt.double(), k)                # exact in float64 for these ranges
    y = (Wt.reshape(Co, -1).double() @ cols)       # (B,Co,L)
    return y.round().long().reshape(B, Co, Hh - k + 1, Ww - k + 1).numpy()


def int_forward(layers, q):
    """q: (B,S,S) integer codes 0..255 -> int logits (B,) -- bit-exact model of
    the RTL datapath (int accumulate, per-channel multiplier+shift requant)."""
    x = q.astype(np.int64)[:, None]
    for d in layers:
        if d["kind"] == "conv":
            acc = _conv_int(x, d["W"]) + d["b"].reshape(1, -1, 1, 1)
        else:
            acc = (x.reshape(x.shape[0], -1).astype(np.float64) @ d["W"].T.astype(np.float64)).round().astype(np.int64) + d["b"]
        if d["relu"]:
            M = d["M"].reshape(1, -1, *([1] * (acc.ndim - 2)))
            y = (acc * M + (1 << (d["S"] - 1))) >> d["S"]
            x = np.clip(y, 0, 255)
        else:
            x = acc
        if d["pool"]:
            B, C, Hh, Ww = x.shape
            x = x[:, :, :Hh // 2 * 2, :Ww // 2 * 2].reshape(B, C, Hh // 2, 2, Ww // 2, 2).max(axis=(3, 5))
    return x.reshape(x.shape[0]).astype(np.float64) * (layers[-1]["s_in"] * layers[-1]["sw"][0])


# ------------------------------------------------------------------- driver
def load(ckpt):
    ck = torch.load(ckpt, weights_only=False)
    cfg = ck["cfg"]
    m = H.build(cfg)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return ck, cfg, m


def pool_codes(x, down):
    """Integer 2x2 (or dxd) average with round-half-up: (sum + d*d/2) // (d*d) --
    exactly what the RTL does (adder tree + add + shift)."""
    if down <= 1:
        return x
    B, Hh, Ww = x.shape
    s = x.reshape(B, Hh // down, down, Ww // down, down).sum(axis=(2, 4))
    return (s + (down * down) // 2) // (down * down)


def raw_codes(d, data_t, idx, size, down=1):
    P = data_t.shape[1]
    o = (P - size) // 2
    x = data_t[idx][:, o:o + size, o:o + size].numpy().astype(np.int64)
    return pool_codes(x, down)


def score_int(layers, d, data_t, idx, size, down=1, chunk=4096):
    out = []
    for i in range(0, len(idx), chunk):
        out.append(int_forward(layers, raw_codes(d, data_t, idx[i:i + chunk], size, down)))
    return np.concatenate(out)


def full_int_scores(layers, d, split, data_t, size, down=1):
    idx = d.idx[split]
    out = np.full(len(idx), H.NEG_FLOOR, dtype=np.float64)
    ok = np.isin(idx, d.pidx[split]) if H.GATE_TAU is not None else np.ones(len(idx), bool)
    out[ok] = score_int(layers, d, data_t, idx[ok], size, down)
    return out


def codes_batch(gx, size, down, dy, dx):
    """gx (B,P,P) float codes on device -> (B,size/down,size/down) float codes, with an
    optional whole-patch shift (edge-replicated) and the RTL's rounded integer pooling."""
    B, P = gx.shape[0], gx.shape[1]
    x = gx
    pad = max(abs(dy), abs(dx))
    if pad:
        x = F.pad(x.unsqueeze(1), (pad, pad, pad, pad), mode="replicate").squeeze(1)
    o = (P - size) // 2 + pad
    x = x[:, o + dy:o + dy + size, o + dx:o + dx + size]
    if down > 1:
        x = torch.floor(F.avg_pool2d(x.unsqueeze(1), down).squeeze(1) + 0.5)
    return x


def export_rtl(ints, cfg, outdir, thr_real, d, data_t, golden_idx):
    """Weights/bias/requant as $readmemh hex + manifest + golden vectors."""
    os.makedirs(outdir, exist_ok=True)

    def hx(v, bits):
        return format(int(v) & ((1 << bits) - 1), "0%dx" % (bits // 4))

    pool_in_data = 2 if (H.VARIANT in ("full32", "hwspec") or H.VARIANT.startswith("pooldet")) else 1            # full32 stores patches already 2x2-pooled
    man = dict(patch_px=int(data_t.shape[1]) * pool_in_data,     # window the hardware must buffer (rows x cols)
               front_pool=pool_in_data * cfg.get("down", 1),     # integer round-half-up average before the net
               net_input=cfg["size"] // cfg.get("down", 1), m_bits=M_BITS, layers=[])
    for i, L in enumerate(ints):
        base = f"L{i}_{L['kind']}"
        with open(os.path.join(outdir, base + "_w.hex"), "w") as f:     # [cout][cin][ky][kx], int8
            for v in L["W"].reshape(-1):
                f.write(hx(v, 8) + "\n")
        with open(os.path.join(outdir, base + "_b.hex"), "w") as f:     # int32 per output channel
            for v in L["b"]:
                f.write(hx(v, 32) + "\n")
        ent = dict(name=base, kind=L["kind"], W_shape=list(L["W"].shape), pool=bool(L["pool"]), relu=bool(L["relu"]))
        if L["relu"]:
            with open(os.path.join(outdir, base + "_m.hex"), "w") as f:  # per-channel requant multiplier
                for v in L["M"]:
                    f.write(hx(v, 16) + "\n")
            ent.update(shift=int(L["S"]), M_max=int(L["M"].max()))
        man["layers"].append(ent)
    last = ints[-1]
    scale = float(last["s_in"] * last["sw"][0])
    man["logit_scale"] = scale
    man["thresholds_int"] = {k: int(np.ceil(v / scale)) for k, v in thr_real.items()}
    man["thresholds_real"] = {k: float(v) for k, v in thr_real.items()}
    # golden vectors: input codes (after the integer down-sample) + expected int32 accumulator
    size, down = cfg["size"], cfg.get("down", 1)
    q = raw_codes(d, data_t, golden_idx, size, down)
    lg = int_forward(ints, q) / scale
    with open(os.path.join(outdir, "golden_in.hex"), "w") as f:
        for patch in q:
            f.write("".join(hx(v, 8) for v in patch.reshape(-1)) + "\n")
    with open(os.path.join(outdir, "golden_logit_int.txt"), "w") as f:
        for v in lg:
            f.write(str(int(round(v))) + "\n")
    man["golden_vectors"] = int(len(golden_idx))
    json.dump(man, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["qat", "eval"])
    ap.add_argument("ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jitter", type=int, default=4)
    ap.add_argument("--export", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ck, cfg, model = load(a.ckpt)
    assert not cfg.get("convs2"), "two-tower export is not implemented"
    d = H.HWData()
    data_t, c1_t = d.tensors()
    size, down = cfg["size"], cfg.get("down", 1)
    layers = fold(model, cfg)
    dev = torch.device("cuda")

    # sanity: folded float net == original (float pooling, no rounding)
    j = d.pidx["val"][:2048]
    with torch.no_grad():
        ref = model(H.prep_cfg(data_t[j], c1_t[j], cfg)).cpu().numpy()
        o = (data_t.shape[1] - size) // 2
        x = data_t[j][:, o:o + size, o:o + size].float().unsqueeze(1)
        if down > 1:
            x = F.avg_pool2d(x, down)
        for l in layers:
            W = torch.tensor(l["W"], dtype=torch.float32); b = torch.tensor(l["b"], dtype=torch.float32)
            x = F.conv2d(x, W, b) if l["kind"] == "conv" else F.linear(x.flatten(1), W, b)
            if l["relu"]: x = F.relu(x)
            if l["pool"]: x = F.max_pool2d(x, 2)
        print(f"BN + input-affine fold: max |err| vs original net = {np.abs(x.squeeze(1).numpy() - ref).max():.2e}", flush=True)

    cal = raw_codes(d, data_t, d.pidx["train"][::37][:20000], size, down)
    fq = FQNet(layers, calibrate(layers, cal)).to(dev)
    tag = os.path.splitext(a.ckpt)[0]

    if a.mode == "qat":
        tr = d.pidx["train"]
        pos = tr[d.labels[tr]]; neg = tr[~d.labels[tr]]
        fr = cfg.get("frag", "all")
        if fr != "all":
            kk = int(fr); key = d.img[pos] * 256 + d.gt[pos]
            order = np.lexsort((-d.area[pos], key)); ks = key[order]
            first = np.r_[True, ks[1:] != ks[:-1]]
            rk = np.arange(len(ks)) - np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
            pos = pos[order[rk < kk]]
        model.to(dev)
        ns = H.score(model, data_t, c1_t, neg, cfg, dev)
        nh = max(int(0.05 * len(neg)), 1000)
        hard = neg[np.argpartition(-ns, nh)[:nh]]
        opt = torch.optim.AdamW(fq.parameters(), lr=a.lr, weight_decay=0.0)
        best, best_state, sched = 1e18, None, None
        ints0 = export_int(fq)                       # PTQ solution is itself a candidate (QAT may not make it worse)
        v0 = H.cascade_curve(d, d.idx["val"], full_int_scores(ints0, d, "val", data_t, size, down))
        best = np.mean([v0[k]["fa_per_img"] for k in ("0.7", "0.75", "0.8", "0.85", "0.9")])
        best_state = {k: v.detach().clone() for k, v in fq.state_dict().items()}
        print(f"[qat] PTQ start: INT8 val mean FA/img(70-90%) {best:.3f}", flush=True)
        for ep in range(a.epochs):
            n_neg = min(cfg["neg_ratio"] * len(pos), len(neg))
            nhd = n_neg // 2
            ids = np.sort(np.concatenate([pos, np.random.choice(hard, nhd), np.random.choice(neg, n_neg - nhd, replace=False)]))
            gx = data_t[ids].to(dev)                                  # uint8, full patch
            gy = torch.from_numpy(d.labels[ids].astype(np.float32)).to(dev)
            perm = torch.randperm(len(ids), device=dev)
            if sched is None:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * ((len(ids) + a.bs - 1) // a.bs))
            fq.train()
            for i in range(0, len(perm), a.bs):
                b = perm[i:i + a.bs]
                xb = gx[b].float()
                m = torch.rand(len(b), 3, device=dev) < 0.5           # dihedral aug on the codes
                xb = torch.where(m[:, 0].view(-1, 1, 1), xb.flip(2), xb)
                xb = torch.where(m[:, 1].view(-1, 1, 1), xb.flip(1), xb)
                xb = torch.where(m[:, 2].view(-1, 1, 1), xb.transpose(1, 2), xb)
                dy = int(np.random.randint(-a.jitter, a.jitter + 1)); dx = int(np.random.randint(-a.jitter, a.jitter + 1))
                x = codes_batch(xb, size, down, dy, dx)
                loss = F.binary_cross_entropy_with_logits(fq(x), gy[b])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            ints = export_int(fq)
            sv = full_int_scores(ints, d, "val", data_t, size, down)
            vc = H.cascade_curve(d, d.idx["val"], sv)
            m90 = np.mean([vc[k]["fa_per_img"] for k in ("0.7", "0.75", "0.8", "0.85", "0.9")])
            flag = ""
            if m90 < best:
                best = m90; best_state = {k: v.detach().clone() for k, v in fq.state_dict().items()}; flag = " *"
            print(f"[qat] ep {ep} loss {loss.item():.4f} | INT8 val mean FA/img(70-90%) {m90:.3f}  "
                  f"@80/85/90: {vc['0.8']['fa_per_img']:.2f} {vc['0.85']['fa_per_img']:.2f} {vc['0.9']['fa_per_img']:.2f}{flag}", flush=True)
        fq.load_state_dict(best_state)

    ints = export_int(fq)
    res = {}
    for split in ("val", "test"):
        sf = H.full_scores(model.to(dev), d, split, data_t, c1_t, cfg, dev)
        si = full_int_scores(ints, d, split, data_t, size, down)
        res[split] = dict(float_curve=H.cascade_curve(d, d.idx[split], sf), int_curve=H.cascade_curve(d, d.idx[split], si))
        np.save(f"{tag}_{a.mode}_int_{split}.npy", si.astype(np.float32))
    sv = np.load(f"{tag}_{a.mode}_int_val.npy").astype(np.float64)
    st = np.load(f"{tag}_{a.mode}_int_test.npy").astype(np.float64)
    thr = {str(r): H.thr_for_retention(d, d.idx["val"], sv, r) for r in H.RETS}
    res["deploy"] = {k: H.at_threshold(d, d.idx["test"], st, t) for k, t in thr.items()}
    print("\nship-retention | FLOAT val/test FA/img | INT8 val/test FA/img")
    for r in H.RETS:
        k = str(r)
        print(f"   {r:.2f}       | {res['val']['float_curve'][k]['fa_per_img']:6.2f} / {res['test']['float_curve'][k]['fa_per_img']:6.2f}"
              f"      | {res['val']['int_curve'][k]['fa_per_img']:6.2f} / {res['test']['int_curve'][k]['fa_per_img']:6.2f}")
    json.dump(res, open(f"{tag}_{a.mode}_int.json", "w"), indent=1)
    torch.save(dict(fq_state=fq.state_dict(), meta=fq.meta, cfg=cfg, ints=ints), f"{tag}_{a.mode}.pt")
    if a.export:
        gi = d.pidx["test"][:256]
        export_rtl(ints, cfg, a.export, thr, d, data_t, gi)
        print("RTL export ->", a.export)
    print("saved", f"{tag}_{a.mode}.pt")


if __name__ == "__main__":
    main()
