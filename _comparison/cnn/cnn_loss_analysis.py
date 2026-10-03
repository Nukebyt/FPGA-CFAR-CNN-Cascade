# -*- coding: utf-8 -*-
"""Which ships does the CNN stage lose (config A, test images)?  usage: HWDATA=pooldetA python cnn_loss_analysis.py <run_name> [target_retention]"""
import os, sys, numpy as np, pandas as pd
import hwlib as H
name = sys.argv[1]; target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.97
d = H.HWData(); m = np.load(os.path.join(H.CACHE, "meta.npz")); gt2 = m["gtIdx2"].astype(np.int64)
sv = np.load(os.path.join(H.RES, "hw", f"{name}_val.npy")).astype(float); st = np.load(os.path.join(H.RES, "hw", f"{name}_test.npy")).astype(float)

def keyed(idx, sc):
    img, g1, g2, lab = d.img[idx], d.gt[idx], gt2[idx], d.labels[idx]
    k1 = (img * 256 + g1)[lab & (g1 > 0)]; s1 = sc[lab & (g1 > 0)]; k2 = (img * 256 + g2)[g2 > 0]; s2 = sc[g2 > 0]
    k = np.concatenate([k1, k2]); s = np.concatenate([s1, s2]); o = np.argsort(k, kind="stable"); k, s = k[o], s[o]
    u, f = np.unique(k, return_index=True)
    return u, np.maximum.reduceat(s, f), np.diff(np.append(f, len(k)))
uv, smv, _ = keyed(d.idx["val"], sv)
thr = float(np.sort(smv)[int(np.floor((1 - target) * len(smv)))])
ut, smt, nev = keyed(d.idx["test"], st)
S = pd.read_csv(os.path.join(H.RES, "pd_study", "phase1_ship_table.csv"), low_memory=False)
S["gt"] = S.groupby("img")["annIdx"].rank(method="first").astype(int)
S["key"] = (S["img"].astype(int) - 1) * 256 + S["gt"]
S = S[S["split"] == "test"].copy()
info = pd.DataFrame({"key": ut, "score": smt, "n_events": nev})
S = S.merge(info, on="key", how="left")
S["delivered"] = S["score"].notna(); S["kept"] = S["score"] >= thr
print(f"{name}: test ships {len(S)}, delivered {S.delivered.mean():.4f}, kept at target {target}: {S.kept.mean():.4f} (thr {thr:.3f})")
lost = S[~S.kept]
print(f"lost {len(lost)}: not delivered {int((~S.delivered).sum())}, rejected by CNN {int((S.delivered & ~S.kept).sum())}")
def tab(col, edges):
    b = pd.cut(S[col].replace(np.inf, 1e9), edges, right=False)
    g = S.groupby(b, observed=True)["kept"].agg(["mean", "size"]); g["lost"] = g["size"] - (g["mean"] * g["size"]).round().astype(int)
    print(f"-- {col}"); print(g.round(3).to_string())
tab("areaMask", [0, 25, 50, 100, 250, 500, 1e9]); tab("dxMax", [-9, 1.0, 1.2, 1.5, 1.8, 9]); tab("nbrDist", [-1, 0.5, 5, 20, 1e9]); tab("distBorder", [-1, 8, 32, 1e9]); tab("n_events", [1, 2, 3, 5, 9, 1e9])
print("-- scene"); print(S.groupby("scene")["kept"].agg(["mean", "size"]).round(3).to_string())
print("score margin of lost-by-CNN ships (score - thr): quantiles", np.round(np.quantile(S.loc[S.delivered & ~S.kept, 'score'] - thr, [.1, .25, .5, .75, .9]), 2))
