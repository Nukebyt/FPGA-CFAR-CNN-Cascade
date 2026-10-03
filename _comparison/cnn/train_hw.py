# -*- coding: utf-8 -*-
"""Train one hardware-constrained CNN discriminator on the full-prevalence,
hardware-geometry candidate stream (see hwlib.py).

Training set: every positive cluster + a negative pool that is half random,
half the currently-hardest negatives (iterative hard-negative mining against
the model's own scores, re-mined every --mine-every epochs).  Priors are NOT
corrected in the loss; the operating threshold is chosen on the validation
split at the TRUE prevalence, which is what a deployed cascade sees.

Model selection is on VALIDATION FA/image @ 90% ship retention.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import hwlib as H

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--convs", default="5-8-1,5-16-1")
ap.add_argument("--fcs", default="32")
ap.add_argument("--size", type=int, default=32)
ap.add_argument("--norm", default="global", choices=["global", "pmean", "c1"])
ap.add_argument("--convs2", default="", help="second (context) tower spec; enables the two-tower net")
ap.add_argument("--size2", type=int, default=64)
ap.add_argument("--down2", type=int, default=2)
ap.add_argument("--down", type=int, default=1, help="2x2 average down-sample of the cropped patch before the network")
ap.add_argument("--jitter", type=int, default=0)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--neg-ratio", type=int, default=16, help="negatives per positive per epoch")
ap.add_argument("--hard-frac", type=float, default=0.5, help="share of epoch negatives drawn from the hard pool")
ap.add_argument("--hard-top", type=float, default=0.05, help="hard pool = top fraction of train negatives by score")
ap.add_argument("--mine-every", type=int, default=5)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--frag", default="all", help="'all' or k: train only on the k largest ship-overlapping fragments per ship (other fragments are ignored, not treated as negatives)")
ap.add_argument("--train-all", action="store_true", help="train on every cluster in the file, not just those passing the deployment gate")
ap.add_argument("--teacher", default="")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--temp", type=float, default=2.0)
ap.add_argument("--out", default=os.path.join(H.RES, "hw"))
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True

cfg = dict(vars(args))
d = H.HWData()
data_t, c1_t = d.tensors()
tr = d.idx["train"] if args.train_all else d.pidx["train"]
tr_pos = tr[d.labels[tr]]
tr_neg = tr[~d.labels[tr]]
if args.frag != "all":
    k = int(args.frag)
    key = d.img[tr_pos] * 256 + d.gt[tr_pos]
    order = np.lexsort((-d.area[tr_pos], key))
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    pos_in_grp = np.arange(len(ks)) - np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
    tr_pos = tr_pos[order[pos_in_grp < k]]
print(f"[{args.name}] train {len(tr)} ({len(tr_pos)} pos / {len(tr_neg)} neg), "
      f"val {len(d.pidx['val'])}/{len(d.idx['val'])}, test {len(d.pidx['test'])}/{len(d.idx['test'])} "
      f"(gated/total; data={H.VARIANT}, P={d.P}, gate={H.GATE_TAU})", flush=True)

model = H.build(cfg).to(dev)
print(f"[{args.name}] params={model.params} macs/patch={model.macs} flat={model.flat}", flush=True)

teacher, tcfg = None, None
if args.teacher:
    ck = torch.load(args.teacher, weights_only=False)
    tcfg = ck["cfg"]
    teacher = H.build(tcfg).to(dev)
    teacher.load_state_dict(ck["state_dict"])
    teacher.eval()

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
steps_per_epoch = None
sched = None


def val_metrics():
    s = H.full_scores(model, d, "val", data_t, c1_t, cfg, dev)
    return H.cascade_curve(d, d.idx["val"], s), s


hard_pool = None
best, best_state, best_ep = 1e18, None, -1
t0 = time.time()
for ep in range(args.epochs):
    # ---- (re)mine hard negatives ------------------------------------------
    if ep > 0 and ep % args.mine_every == 0:
        ns = H.score(model, data_t, c1_t, tr_neg, cfg, dev)
        k = max(int(args.hard_top * len(tr_neg)), 1000)
        hard_pool = tr_neg[np.argpartition(-ns, k)[:k]]
    n_neg = min(args.neg_ratio * len(tr_pos), len(tr_neg))
    if hard_pool is None:
        negs = np.random.choice(tr_neg, n_neg, replace=False)
    else:
        n_h = int(args.hard_frac * n_neg)
        negs = np.concatenate([np.random.choice(hard_pool, n_h, replace=True),
                               np.random.choice(tr_neg, n_neg - n_h, replace=False)])
    ids = np.concatenate([tr_pos, negs])
    np.random.shuffle(ids)
    ids_s = np.sort(ids)           # sorted gather is faster; shuffle again on GPU
    gx = data_t[ids_s].to(dev)
    gc = c1_t[ids_s].to(dev)
    gy = torch.from_numpy(d.labels[ids_s].astype(np.float32)).to(dev)
    perm = torch.randperm(len(ids_s), device=dev)

    if sched is None:
        nb = (len(ids_s) + args.bs - 1) // args.bs
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * nb, pct_start=0.1)

    model.train()
    tot = 0.0
    for i in range(0, len(perm), args.bs):
        b = perm[i:i + args.bs]
        x = H.prep_cfg(gx[b], gc[b], cfg, args.jitter, train=True)
        y = gy[b]
        z = model(x)
        loss = F.binary_cross_entropy_with_logits(z, y)
        if teacher is not None:
            with torch.no_grad():
                xt = H.prep_cfg(gx[b], gc[b], tcfg)
                zt = teacher(xt)
            T = args.temp
            ls = F.binary_cross_entropy_with_logits(z / T, torch.sigmoid(zt / T)) * T * T
            loss = (1 - args.alpha) * loss + args.alpha * ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched.last_epoch < sched.total_steps - 1:
            sched.step()
        tot += loss.item() * len(b)
    del gx, gc, gy

    if ep >= args.epochs // 3 or ep == 0:
        vc, _ = val_metrics()
        m = vc["0.9"]["fa_per_img"]
        flag = ""
        if m < best:
            best, best_ep = m, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(dict(state_dict=best_state, cfg=cfg, epoch=ep), os.path.join(args.out, args.name + "_best.pt"))
            flag = " *"
        print(f"[{args.name}] ep {ep:2d} loss {tot/len(perm):.4f} | val FA/img @ret 80/85/90/95: "
              f"{vc['0.8']['fa_per_img']:.1f} {vc['0.85']['fa_per_img']:.1f} {m:.1f} "
              f"{vc['0.95']['fa_per_img']:.1f}  ({time.time()-t0:.0f}s){flag}", flush=True)
    else:
        print(f"[{args.name}] ep {ep:2d} loss {tot/len(perm):.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---- final: best-val checkpoint -> val-selected thresholds applied to test ----
model.load_state_dict(best_state)
sv = H.full_scores(model, d, "val", data_t, c1_t, cfg, dev)
st = H.full_scores(model, d, "test", data_t, c1_t, cfg, dev)
res = dict(name=args.name, cfg=cfg, params=model.params, macs=model.macs, best_epoch=best_ep,
           val_curve=H.cascade_curve(d, d.idx["val"], sv),
           test_curve=H.cascade_curve(d, d.idx["test"], st),
           val_ap=H.ap_score(d, d.idx["val"], sv), test_ap=H.ap_score(d, d.idx["test"], st),
           deploy={})
for r in H.RETS:
    thr = H.thr_for_retention(d, d.idx["val"], sv, r)
    res["deploy"][str(r)] = H.at_threshold(d, d.idx["test"], st, thr)
np.save(os.path.join(args.out, args.name + "_val.npy"), sv.astype(np.float32))
np.save(os.path.join(args.out, args.name + "_test.npy"), st.astype(np.float32))
torch.save(dict(state_dict=model.state_dict(), cfg=cfg, res=res), os.path.join(args.out, args.name + ".pt"))
res["shift_test"] = {}
for sh in ((2, 2), (3, 3)):          # trigger-offset robustness: whole-patch shift, test split
    ss = H.full_scores(model, d, "test", data_t, c1_t, cfg, dev, shift=sh)
    cc = H.cascade_curve(d, d.idx["test"], ss)
    res["shift_test"][f"{sh[0]},{sh[1]}"] = {k: v["fa_per_img"] for k, v in cc.items()}
with open(os.path.join(args.out, "results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
tc = res["test_curve"]
print(f"[{args.name}] DONE params={model.params} macs={model.macs} best_ep={best_ep} "
      f"VAL FA@90={res['val_curve']['0.9']['fa_per_img']:.1f} | TEST FA/img @ret 80/85/90/95: "
      f"{tc['0.8']['fa_per_img']:.1f} {tc['0.85']['fa_per_img']:.1f} {tc['0.9']['fa_per_img']:.1f} "
      f"{tc['0.95']['fa_per_img']:.1f}  AP={res['test_ap']:.4f}", flush=True)
