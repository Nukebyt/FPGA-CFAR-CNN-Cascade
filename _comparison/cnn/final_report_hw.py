# -*- coding: utf-8 -*-
"""Paper-2 funnel table for the two exported INT8 models: thresholds chosen on
VALIDATION at a target ship retention, applied unchanged to the held-out TEST
split; plus bootstrap CIs over test images and the hwcost budget."""
import json
import os
import numpy as np

os.environ.setdefault("HWDATA", "full32"); os.environ.setdefault("HWGATE", "0.45")
import hwlib as H
import hwcost as C

d = H.HWData()
MODELS = json.loads(os.environ.get("MODELS", '{"DEEP": "f_deep_s1", "SMALL": "f_small_s1"}'))
PATCH_PX = 64
RET_TARGETS = (0.80, 0.85, 0.90)
n_test = len(d.split_imgs["test"])
n_ships_test = int(d.nships[d.split_imgs["test"]].sum())
idx = d.idx["test"]
pos, order, start = H.ship_table(d, idx)
n_reach = len(start)
cand_cfar = d.cand_per_img["test"]
gate_pass = len(d.pidx["test"]) / n_test
print(f"TEST: {n_test} images, {n_ships_test} GT ships, {n_reach} CFAR-reachable ({100*n_reach/n_ships_test:.1f}%)")
print(f"CFAR (Weibull sli=17/guard=13, Pfa=1e-3) alone: {cand_cfar:.0f} candidates/img "
      f"({cand_cfar - d.labels[idx].sum()/n_test:.0f} false alarms/img; ship-overlapping clusters are all retained in the file)")
print(f"after free gate (3x3 centre mean - c1 >= {H.GATE_TAU}): {gate_pass:.0f} candidates/img reach the CNN "
      f"({100*gate_pass/cand_cfar:.1f}%)\n")

out = {"test_images": n_test, "test_ships": n_ships_test, "reachable": n_reach, "cfar_cand_per_img": cand_cfar,
       "gate_cand_per_img": gate_pass, "models": {}}


def boot_fa(scores, thr, B=300, seed=0):
    """CI of FA/img and of ship retention at a FIXED threshold, resampling test images."""
    rng = np.random.RandomState(seed)
    imgs = d.split_imgs["test"]
    lab = d.labels[idx]
    img = d.img[idx]
    acc = scores >= thr
    fa_per = np.bincount(img[~lab & acc], minlength=len(d.nships)).astype(float)
    # ship retained flags per (img,gt)
    key = (img * 256 + d.gt[idx])[lab]
    o = np.argsort(key, kind="stable"); ks = key[o]
    u, st = np.unique(ks, return_index=True)
    smax = np.maximum.reduceat(scores[lab][o], st)
    ship_img = (u // 256).astype(int)
    kept = (smax >= thr).astype(float)
    ships_per = np.bincount(ship_img, minlength=len(d.nships)).astype(float)
    kept_per = np.bincount(ship_img, weights=kept, minlength=len(d.nships))
    fa, ret = [], []
    for _ in range(B):
        s = rng.choice(imgs, len(imgs), replace=True)
        fa.append(fa_per[s].sum() / len(s)); ret.append(kept_per[s].sum() / ships_per[s].sum())
    return np.percentile(fa, [2.5, 97.5]), np.percentile(ret, [2.5, 97.5])


for name, run in MODELS.items():
    tag = H.RES + rf"\hw\{run}_qat"
    sv = np.load(tag + "_int_val.npy").astype(np.float64)
    st = np.load(tag + "_int_test.npy").astype(np.float64)
    ck = __import__("torch").load(H.RES + rf"\hw\{run}.pt", weights_only=False)
    cfg = ck["cfg"]
    hw = C.report(name, cfg, lanes=128, cand_per_img=gate_pass, patch_px=PATCH_PX)
    hw297 = C.report(name + " @297 lanes", cfg, lanes=297, cand_per_img=gate_pass, verbose=False, patch_px=PATCH_PX)
    out["models"][name] = dict(run=run, params=hw["params"], macs=hw["macs"], hw=hw, hw297=hw297, ops={})
    print(f"\n{name} INT8 (val-selected thresholds -> test)")
    print(f"  {'target':>6} {'test ship ret':>13} {'[95% CI]':>15} {'FA/img':>7} {'[95% CI]':>13} {'cand/img out':>12} "
          f"{'reduction':>10} {'e2e ship recall':>16}")
    for r in RET_TARGETS:
        thr = H.thr_for_retention(d, d.idx["val"], sv, r)
        m = H.at_threshold(d, idx, st, thr)
        fa_ci, ret_ci = boot_fa(st, thr)
        e2e = m["ship_retention"] * n_reach / n_ships_test
        print(f"  {r:6.2f} {100*m['ship_retention']:12.1f}% [{100*ret_ci[0]:5.1f},{100*ret_ci[1]:5.1f}] "
              f"{m['fa_per_img']:7.2f} [{fa_ci[0]:4.2f},{fa_ci[1]:5.2f}] {m['cand_per_img']:12.2f} "
              f"{cand_cfar/m['cand_per_img']:9.0f}x {100*e2e:15.1f}%")
        out["models"][name]["ops"][str(r)] = dict(**m, fa_ci=fa_ci.tolist(), ret_ci=ret_ci.tolist(), e2e_ship_recall=e2e,
                                                   reduction_vs_cfar=cand_cfar / m["cand_per_img"])
json.dump(out, open(os.path.join(H.RES, "hw", f"final_funnel_{H.VARIANT}.json"), "w"), indent=1)
print(f"\nsaved Results/hw/final_funnel_{H.VARIANT}.json")
