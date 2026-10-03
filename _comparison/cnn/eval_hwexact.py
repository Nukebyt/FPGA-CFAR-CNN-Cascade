# -*- coding: utf-8 -*-
"""Score the exported INT8 models on HARDWARE-EXACT test events (bit-exact fixed-point Weibull detection,
NMS trigger, hardware gate, pooled-store patches -- extract_cnn_patches_hwspec('Fixed',true)) and compare with
the float-pipeline test split the models were trained/validated against.

Operating thresholds are chosen on the float-pipeline VALIDATION split (as in final_report_hw.py) and applied
unchanged to both test sets.  usage:  HWDATA=hwspec HWGATE=0.75 python eval_hwexact.py  [model names...]
"""
import json
import os
import sys
import types
import numpy as np
import h5py
import torch

os.environ.setdefault("HWDATA", "hwspec"); os.environ.setdefault("HWGATE", "0.75")
import hwlib as H
import quant_hw as Q

RUNS = json.loads(os.environ.get("MODELS", '{"XL": "h_xl_s1", "DEEP": "h_deep_s1", "SMALL": "h_small_s1"}'))
RET = (0.80, 0.85, 0.90)
GATE = H.GATE_TAU
d = H.HWData()                                    # float-pipeline splits (for val thresholds + reference test)

# ---- hardware-exact test events -------------------------------------------------------------------
with h5py.File(os.path.join(H.RES, "cnn_patches_hwexact_test.mat"), "r") as f:
    px = np.array(f["patches"])
    lab = np.array(f["labels"]).squeeze().astype(bool)
    img = np.array(f["imgIdx"]).squeeze().astype(np.int64) - 1
    gt = np.array(f["gtIdx"]).squeeze().astype(np.int64)
    gate = np.array(f["gate"]).squeeze().astype(np.float64)
    ntrig = np.array(f["imgNTrig"]).squeeze().astype(np.float64)
    nships = np.array(f["imgNShips"]).squeeze().astype(np.int64)
test_imgs = np.sort(np.loadtxt(os.path.join(H.RES, "test_img_idx.txt")).astype(np.int64) - 1)
X = types.SimpleNamespace(labels=lab, img=img, gt=gt, nships=nships, split_imgs={"test": test_imgs},
                          idx={"test": np.arange(len(lab))}, cand_per_img={"test": float(ntrig[test_imgs].mean())})
n_img = len(test_imgs)
pos, order, start = H.ship_table(X, X.idx["test"])
print(f"hardware-exact test: {n_img} images, {int(nships[test_imgs].sum())} GT ships, {len(start)} reachable "
      f"({100*len(start)/nships[test_imgs].sum():.1f}%), {X.cand_per_img['test']:.0f} triggers/img, "
      f"{(gate >= GATE).sum()/n_img:.0f} gated events/img")
print(f"float-pipeline test:  {len(d.split_imgs['test'])} images (same images), "
      f"{d.cand_per_img['test']:.0f} triggers/img, {len(d.pidx['test'])/len(d.split_imgs['test']):.0f} gated events/img")

ok = gate >= GATE


def int_scores(ints, patches):
    out = np.empty(len(patches))
    for i in range(0, len(patches), 4096):
        out[i:i + 4096] = Q.int_forward(ints, patches[i:i + 4096].astype(np.int64))
    return out


def boot(scores, thr, ds, split_imgs, B=300, seed=0):
    rng = np.random.RandomState(seed)
    lab_, img_, gt_ = ds.labels, ds.img, ds.gt
    acc = scores >= thr
    n_all = len(ds.nships)
    fa_per = np.bincount(img_[~lab_ & acc], minlength=n_all).astype(float)
    key = (img_ * 256 + gt_)[lab_]
    o = np.argsort(key, kind="stable"); ks = key[o]
    u, st = np.unique(ks, return_index=True)
    smax = np.maximum.reduceat(scores[lab_][o], st)
    ship_img = (u // 256).astype(int)
    kept_per = np.bincount(ship_img, weights=(smax >= thr).astype(float), minlength=n_all)
    ships_per = np.bincount(ship_img, minlength=n_all).astype(float)
    fa, ret = [], []
    for _ in range(B):
        s = rng.choice(split_imgs, len(split_imgs), replace=True)
        fa.append(fa_per[s].sum() / len(s)); ret.append(kept_per[s].sum() / ships_per[s].sum())
    return np.percentile(fa, [2.5, 97.5]), np.percentile(ret, [2.5, 97.5])


out = {}
for name, run in RUNS.items():
    ck = torch.load(os.path.join(H.RES, "hw", f"{run}_qat.pt"), weights_only=False)
    ints = ck["ints"]
    sv = np.load(os.path.join(H.RES, "hw", f"{run}_qat_int_val.npy")).astype(np.float64)
    st_float = np.load(os.path.join(H.RES, "hw", f"{run}_qat_int_test.npy")).astype(np.float64)
    s_exact = np.full(len(lab), H.NEG_FLOOR)
    s_exact[ok] = int_scores(ints, px[ok])
    out[name] = {}
    print(f"\n{name} INT8  (thresholds from float-pipeline VALIDATION)")
    print(f"  {'target':>6} | {'float-pipeline test: ret    FA/img':>38} | {'hardware-exact test: ret [95% CI]      FA/img [95% CI]':>56} | cand/img out")
    for r in RET:
        thr = H.thr_for_retention(d, d.idx["val"], sv, r)
        mf = H.at_threshold(d, d.idx["test"], st_float, thr)
        me = H.at_threshold(X, X.idx["test"], s_exact, thr)
        fa_ci, ret_ci = boot(s_exact, thr, X, test_imgs)
        print(f"  {r:6.2f} | {100*mf['ship_retention']:25.1f}%  {mf['fa_per_img']:8.2f} | "
              f"{100*me['ship_retention']:8.1f}% [{100*ret_ci[0]:5.1f},{100*ret_ci[1]:5.1f}]   {me['fa_per_img']:6.2f} [{fa_ci[0]:4.2f},{fa_ci[1]:5.2f}] | "
              f"{me['cand_per_img']:6.1f}")
        out[name][str(r)] = dict(thr=thr, float_pipeline=mf, hw_exact=me, hw_fa_ci=fa_ci.tolist(), hw_ret_ci=ret_ci.tolist())
jp = os.path.join(H.RES, "hw", "eval_hwexact.json")
prev = json.load(open(jp)) if os.path.exists(jp) else {}
prev.update(out)                                  # merge: keep models evaluated in earlier invocations
json.dump(prev, open(jp, "w"), indent=1, default=float)
print("\nsaved Results/hw/eval_hwexact.json")
