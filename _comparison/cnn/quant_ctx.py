# -*- coding: utf-8 -*-
"""Independent numpy reference (bit-exact model of cnn_core_ctx.v) for the context network written by train_q3.py, and golden-vector export for the RTL testbench.
   python quant_ctx.py <name_int3.pt> <out_dir> [n=48]      (golden vectors from validation events of the hardware-exact candidate set)"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def _conv(x, W):
    xt = torch.from_numpy(x); Wt = torch.from_numpy(W)
    B, C, Hh, Ww = xt.shape; Co, _, k, _ = Wt.shape
    cols = F.unfold(xt.double(), k)
    y = Wt.reshape(Co, -1).double() @ cols
    return y.round().long().reshape(B, Co, Hh - k + 1, Ww - k + 1).numpy()


def _req(acc, L):
    M = np.asarray(L["M"], np.int64).reshape(1, -1, *([1] * (acc.ndim - 2)))
    return np.clip((acc * M + (1 << (L["S"] - 1))) >> L["S"], 0, 255)


def _tower(x, layers):
    x = x.astype(np.int64)[:, None]
    for L in layers:
        x = _req(_conv(x, np.asarray(L["W"], np.int64)) + np.asarray(L["b"], np.int64).reshape(1, -1, 1, 1), L)
        if L["pool"]:
            B, C, Hh, Ww = x.shape
            x = x[:, :, :Hh // 2 * 2, :Ww // 2 * 2].reshape(B, C, Hh // 2, 2, Ww // 2, 2).max(axis=(3, 5))
    return x.reshape(x.shape[0], -1)


def _fc(x, L):
    return (x.astype(np.float64) @ np.asarray(L["W"], np.float64).T).round().astype(np.int64) + np.asarray(L["b"], np.int64)


def int_forward3(ints, xf, xc, xs):
    f = _tower(xf, ints["fine"]); c = _tower(xc, ints["ctx"])
    sv = _req(_fc(xs.astype(np.int64), ints["side"]), ints["side"])
    h = _req(_fc(np.concatenate([f, c, sv], 1), ints["h1"]), ints["h1"])
    return _fc(h, ints["h2"]).reshape(-1)


def load(path):
    return torch.load(path, weights_only=False)["ints3"]


if __name__ == "__main__":
    ints = load(sys.argv[1]); out = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("HWDATA", "pooldetfxA")
    import hwlib as H
    d = H.HWData(); cd = os.environ["CTXDIR"]
    CTX = np.load(os.path.join(cd, "ctx.npy"), mmap_mode="r"); SIDE = np.load(os.path.join(cd, os.environ.get("SIDEFILE", "side_codes.npy")))
    rng = np.random.RandomState(11)
    idx = np.sort(rng.choice(d.idx["val"], n, replace=False))
    xf = np.ascontiguousarray(d.patches[idx]).astype(np.int64); xc = np.ascontiguousarray(CTX[idx]).astype(np.int64); xs = SIDE[idx].astype(np.int64)
    lg = int_forward3(ints, xf, xc, xs)
    with open(os.path.join(out, "gin_fine.hex"), "w") as f1, open(os.path.join(out, "gin_ctx.hex"), "w") as f2, open(os.path.join(out, "gin_side.hex"), "w") as f3, open(os.path.join(out, "golden_logit.txt"), "w") as f4:
        for k in range(n):
            f1.write("".join("%02x" % v for v in xf[k].reshape(-1)) + "\n"); f2.write("".join("%02x" % v for v in xc[k].reshape(-1)) + "\n")
            f3.write("".join("%02x" % v for v in xs[k]) + "\n"); f4.write("%d\n" % lg[k])
    print("golden vectors", n, "logit range", lg.min(), lg.max())
