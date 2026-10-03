# -*- coding: utf-8 -*-
"""Make an exported INT8 model consume the patch in the layout the RTL streams it.

THE BUG THIS FIXES (found 2026-10-03 when the first real frames ran on the FPGA):
  The CNN was trained on the arrays read from the MATLAB -v7.3 patch files with h5py.  MATLAB writes a 32x32xN array
  pat(r,c,n) with reversed dimensions, so the training input is X[n][c][r] = P(r,c): the pooled-store window TRANSPOSED.
  patch_fetch.v streams the window row-major (row r outer, column c inner), i.e. it feeds P(r,c) -- the transpose of what
  the network was trained on.  (cnn_core was verified bit-exact against the Python integer model on golden vectors in the
  training layout, and the cascade against a reference that cut the window row-major, so neither check could see it.)

THE FIX (no retraining, no RTL change): a CNN applied to the transposed input equals the same CNN with
  * every conv kernel transposed in (ky, kx),
  * the first FC after the last conv re-indexed the same way (flatten (C,H,W) -> transpose H,W per channel),
  (2x2 max-pool and per-channel requantisation are transpose-symmetric).  This is an index permutation of integer
  weights, so it is exact: int_forward(ints_T, P) == int_forward(ints, P.T) bit for bit.  This script
    1. writes <out_export_dir> = a copy of <export_dir> with transposed *_w.hex and transposed golden_in.hex
       (the golden logits are unchanged: golden_logit_int.txt is identical),
    2. writes <out_pt> = {"ints": ints_T} for the Python references (sweep250.py, tb/check_cascade.py),
    3. proves the equivalence on the 256 golden vectors and on random patches.

usage: python transpose_export.py <ints.pt> <export_dir> <out_export_dir> <out_pt>
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_hw as Q                                         # noqa: E402


def transpose_ints(ints, net_input=32):
    out, cur_c, cur_h = [], 1, net_input
    for L in ints:
        d = dict(L)
        W = np.asarray(L["W"])
        if L["kind"] == "conv":
            d["W"] = np.ascontiguousarray(W.transpose(0, 1, 3, 2))
            cur_c, k = W.shape[0], W.shape[2]
            cur_h = cur_h - k + 1
        else:
            if cur_h > 1:                                    # FC fed by a feature map: flatten order (C,H,W)
                cout, cflat = W.shape
                hw = int(round(np.sqrt(cflat // cur_c)))
                assert cur_c * hw * hw == cflat, (cur_c, hw, cflat)
                d["W"] = np.ascontiguousarray(W.reshape(cout, cur_c, hw, hw).transpose(0, 1, 3, 2).reshape(cout, cflat))
                cur_h = 1
            cur_c = W.shape[0]
        if L["pool"]:
            cur_h //= 2
        out.append(d)
    return out


def rd_hex_bytes(path):
    return [int(l, 16) for l in open(path) if l.strip()]


def main():
    ints_pt, exp_dir, out_dir, out_pt = sys.argv[1:5]
    ck = torch.load(ints_pt, weights_only=False)
    ints = ck["ints"]
    ints_T = transpose_ints(ints)
    torch.save({"ints": ints_T}, out_pt)

    man = json.load(open(os.path.join(exp_dir, "manifest.json")))
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(exp_dir):
        shutil.copy(os.path.join(exp_dir, fn), os.path.join(out_dir, fn))
    for ent, L, LT in zip(man["layers"], ints, ints_T):
        flat = np.asarray(LT["W"]).reshape(-1)
        assert flat.size == np.prod(ent["W_shape"])
        ref = np.array([v - 256 if v >= 128 else v for v in rd_hex_bytes(os.path.join(exp_dir, ent["name"] + "_w.hex"))])
        assert np.array_equal(ref, np.asarray(L["W"]).reshape(-1)), f"{ent['name']}: ints.pt weights differ from the exported hex"
        with open(os.path.join(out_dir, ent["name"] + "_w.hex"), "w") as f:
            for v in flat:
                f.write("%02x\n" % (int(v) & 0xFF))

    # golden vectors: the core is now fed the window row-major, i.e. the transpose of the training-layout patch
    gin = [l.strip() for l in open(os.path.join(exp_dir, "golden_in.hex")) if l.strip()]
    glog = [int(l) for l in open(os.path.join(exp_dir, "golden_logit_int.txt")) if l.strip()]
    pats = np.array([[int(l[2 * i:2 * i + 2], 16) for i in range(1024)] for l in gin], dtype=np.int64).reshape(-1, 32, 32)
    with open(os.path.join(out_dir, "golden_in.hex"), "w") as f:
        for p in pats:
            f.write("".join("%02x" % v for v in p.T.reshape(-1)) + "\n")

    # ---- proof of equivalence
    s = ints[-1]["s_in"] * ints[-1]["sw"][0]
    a = np.rint(Q.int_forward(ints, pats) / s).astype(np.int64)                 # original model, training layout
    b = np.rint(Q.int_forward(ints_T, pats.transpose(0, 2, 1)) / s).astype(np.int64)   # transposed model, RTL layout
    assert np.array_equal(a, np.array(glog)), "original model does not reproduce the golden logits"
    assert np.array_equal(a, b), "transposed model differs from the original!"
    rng = np.random.RandomState(0)
    r = rng.randint(0, 256, (512, 32, 32)).astype(np.int64)
    assert np.array_equal(Q.int_forward(ints, r), Q.int_forward(ints_T, r.transpose(0, 2, 1))), "random-patch equivalence failed"
    print(f"OK: {len(pats)} golden + 512 random patches: int_forward(ints_T, P) == int_forward(ints, P^T) bit-exactly")
    print("wrote", out_dir, "and", out_pt)


if __name__ == "__main__":
    main()
