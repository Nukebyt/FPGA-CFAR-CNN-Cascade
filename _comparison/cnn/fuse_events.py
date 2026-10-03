# -*- coding: utf-8 -*-
"""Step 2: ship-level evidence fusion.  A second-stage model re-scores every event using the CNN scores of the events around it (a ship usually produces
several events, false alarms tend to be isolated).  Trained on VALIDATION events only (image-grouped 5-fold CV for the validation scores, one fit on all
validation events for the test scores), so the test split stays untouched.
usage: HWDATA=pooldetA python fuse_events.py <run_name>      -> Results/hw/<run_name>_fused_{val,test}.npy ; then  python eval_pooldet.py <run_name> _fused"""
import os, sys, numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
import hwlib as H
name = sys.argv[1]
d = H.HWData(); m = np.load(os.path.join(H.CACHE, "meta.npz")); cxy = np.array(m["cxy"]).T            # (N,2) full-res (y,x)
import glob, h5py
side = np.concatenate([np.array(h5py.File(f, "r")["side"]).T for f in sorted(glob.glob(os.path.join(H.RES, "cnn_ctxA_p*.mat")))]).astype(np.float32)
sv = np.load(os.path.join(H.RES, "hw", f"{name}_val.npy")).astype(np.float64); st = np.load(os.path.join(H.RES, "hw", f"{name}_test.npy")).astype(np.float64)


def feats(idx, sc):
    img = d.img[idx]; F = np.zeros((len(idx), 22), np.float32)
    o = np.argsort(img, kind="stable"); starts = np.r_[0, np.nonzero(np.diff(img[o]))[0] + 1, len(o)]
    for a, b in zip(starts[:-1], starts[1:]):
        ii = o[a:b]; s = sc[ii]; p = cxy[idx[ii]]; n = len(ii)
        tree = cKDTree(p); rank = (-s).argsort().argsort() / max(n - 1, 1)
        col = 0
        F[ii, 0] = s; F[ii, 1] = rank; F[ii, 2] = np.log1p(n); F[ii, 3] = s.max(); F[ii, 4] = side[idx[ii], 0]; F[ii, 5] = side[idx[ii], 1]
        for R in (8, 16, 32, 64):
            nb = tree.query_ball_point(p, R)
            base = 6 + (R // 16 if R > 8 else 0) * 0
            k = {8: 6, 16: 10, 32: 14, 64: 18}[R]
            for j, lst in enumerate(nb):
                lst = [q for q in lst if q != j]
                if lst:
                    v = np.sort(s[lst])[::-1]
                    F[ii[j], k] = len(lst); F[ii[j], k + 1] = v[0]; F[ii[j], k + 2] = v[:3].mean(); F[ii[j], k + 3] = s[j] - v[0]
                else:
                    F[ii[j], k] = 0; F[ii[j], k + 1] = -20; F[ii[j], k + 2] = -20; F[ii[j], k + 3] = s[j] + 20
    return F


vi, ti = d.idx["val"], d.idx["test"]
Fv, Ft = feats(vi, sv), feats(ti, st)
yv = d.labels[vi].astype(int)
oof = np.zeros(len(vi)); grp = d.img[vi]
for trn, tst in GroupKFold(5).split(Fv, yv, grp):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(Fv[trn], yv[trn])
    oof[tst] = clf.predict_proba(Fv[tst])[:, 1]
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(Fv, yv)
pt = clf.predict_proba(Ft)[:, 1]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
np.save(os.path.join(H.RES, "hw", f"{name}_fused_val.npy"), logit(oof)); np.save(os.path.join(H.RES, "hw", f"{name}_fused_test.npy"), logit(pt))
print("saved fused scores for", name)
