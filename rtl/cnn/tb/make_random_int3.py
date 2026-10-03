# -*- coding: utf-8 -*-
"""Random (untrained) integer context network with the real layer shapes, golden vectors computed with the numpy reference quant_ctx.int_forward3.
Purpose: verify cnn_core_ctx.v / gen_cnn_rtl_ctx.py independently of training.   usage: python make_random_int3.py <out_dir> [n=24]"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, r"F:\Projects\CFAR\_comparison\cnn")
import quant_ctx as QC
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
os.makedirs(out, exist_ok=True)
rng = np.random.RandomState(5)
def layer(kind, cout, cin, k, pool, relu, S=None, wscale=40):
    shp = (cout, cin, k, k) if kind == "conv" else (cout, cin)
    W = np.clip(np.round(rng.randn(*shp) * wscale), -127, 127).astype(np.int64)
    d = dict(kind=kind, pool=pool, relu=relu, W=W, b=rng.randint(-2000, 2000, cout).astype(np.int64))
    if relu:
        fan = int(np.prod(shp[1:]))
        S_ = S if S is not None else 18
        d["S"] = S_; d["M"] = rng.randint(1000, 30000, cout).astype(np.int64)
    d["s_in"] = 1.0; d["sw"] = np.ones(cout)
    return d
ints = dict(
    fine=[layer("conv", 16, 1, 5, True, True, 19), layer("conv", 32, 16, 3, False, True, 21), layer("conv", 32, 32, 3, True, True, 22)],
    ctx=[layer("conv", 8, 1, 5, True, True, 19), layer("conv", 16, 8, 3, True, True, 20), layer("conv", 16, 16, 3, True, True, 21)],
    side=layer("fc", 16, 9, 1, False, True, 15, 12),
    h1=layer("fc", 64, 880, 1, False, True, 22, 12), h2=layer("fc", 1, 64, 1, False, False))
torch.save({"ints3": ints}, os.path.join(out, "rnd_int3.pt"))
xf = rng.randint(0, 256, (n, 32, 32)); xc = rng.randint(0, 256, (n, 32, 32)); xs = rng.randint(0, 256, (n, 9))
lg = QC.int_forward3(ints, xf, xc, xs)
with open(os.path.join(out, "gin_fine.hex"), "w") as f1, open(os.path.join(out, "gin_ctx.hex"), "w") as f2, open(os.path.join(out, "gin_side.hex"), "w") as f3, open(os.path.join(out, "golden_logit.txt"), "w") as f4:
    for k in range(n):
        f1.write("".join("%02x" % v for v in xf[k].reshape(-1)) + "\n"); f2.write("".join("%02x" % v for v in xc[k].reshape(-1)) + "\n")
        f3.write("".join("%02x" % v for v in xs[k]) + "\n"); f4.write("%d\n" % lg[k])
print("random model: logits", lg.min(), lg.max(), " std", lg.std())
