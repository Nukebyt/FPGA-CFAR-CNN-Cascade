# -*- coding: utf-8 -*-
"""Export the INT8 model written by train_q.py as RTL inputs (weights / bias / requant hex + manifest + golden vectors).
The pooldetfxA patches are cut row-major from the pooled store (the order patch_fetch.v streams them), so NO transposition is needed
(cf. transpose_export.py, which was only for the older MATLAB-extracted training sets).
usage: HWDATA=pooldetfxA python export_q8.py <name> <out_dir> [n_golden=64]
then:  python ../../rtl/cnn/gen_cnn_rtl_q4.py <out_dir> <rtl_out> --lanes 32"""
import os
import sys

import numpy as np
import torch

os.environ.setdefault("HWDATA", "pooldetfxA")
import hwlib as H
import quant_hw as Q

name, out_dir = sys.argv[1], sys.argv[2]
ng = int(sys.argv[3]) if len(sys.argv) > 3 else 64
ints = torch.load(os.path.join(H.RES, "hw", name + "_int.pt"), weights_only=False)["ints"]
d = H.HWData()
rng = np.random.RandomState(7)
va = d.idx["val"]
gi = np.sort(rng.choice(va, ng, replace=False))
data_t = torch.from_numpy(np.ascontiguousarray(d.patches[gi]))
cfg = {"size": 32, "down": 1}
Q.export_rtl(ints, cfg, out_dir, {"theta0": 0.0}, d, data_t, np.arange(ng))
print("exported", out_dir, "golden vectors", ng)
