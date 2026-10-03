# -*- coding: utf-8 -*-
"""Weibull-only (pooled prescreen) Pd over the ENTIRE dataset: per ship / per image / per split / per scene, from the extracted candidate set.
usage: HWDATA=pooldetA python prescreen_fullsweep.py"""
import os, json, numpy as np, pandas as pd
import hwlib as H
d = H.HWData(); m = np.load(os.path.join(H.CACHE, "meta.npz")); gt2 = m["gtIdx2"].astype(np.int64)
n_img = len(d.nships)
lab = d.labels
k1 = (d.img * 256 + d.gt)[lab & (d.gt > 0)]; k2 = (d.img * 256 + gt2)[gt2 > 0]
deliv = set(np.concatenate([k1, k2]).tolist())
S = pd.read_csv(os.path.join(H.RES, "pd_study", "phase1_ship_table.csv"), low_memory=False)
S["gt"] = S.groupby("img")["annIdx"].rank(method="first").astype(int)
S["key"] = (S["img"].astype(int) - 1) * 256 + S["gt"]
S["delivered"] = S["key"].isin(deliv)
ev_per_img = np.bincount(d.img, minlength=n_img)
print(f"ALL {len(S)} ships / {n_img} images: prescreen Pd (event on the ship) = {S.delivered.mean():.5f}  ({int((~S.delivered).sum())} ships missed)")
for sp in ("train", "val", "test"):
    g = S[S.split == sp]; print(f"  {sp:5s}: {len(g):6d} ships  Pd {g.delivered.mean():.5f}  missed {int((~g.delivered).sum())}")
for sc in ("inshore", "offshore"):
    g = S[S.scene == sc]; print(f"  {sc:8s}: {len(g):6d} ships  Pd {g.delivered.mean():.5f}  missed {int((~g.delivered).sum())}")
im = S.groupby("img").agg(ships=("delivered", "size"), found=("delivered", "sum"), name=("name", "first"), scene=("scene", "first"), split=("split", "first")).reset_index()
im["pd"] = im.found / im.ships; im["events"] = ev_per_img[im.img.astype(int) - 1]
print(f"\nper-image Pd over {len(im)} images with ships: images with Pd = 1: {(im.pd == 1).sum()} ({100*(im.pd==1).mean():.2f} %), Pd >= 0.99: {(im.pd >= 0.99).sum()}, "
      f"images with at least one ship missed: {(im.pd < 1).sum()} ({100*(im.pd<1).mean():.2f} %)")
print("per-image Pd quantiles (0,1,5,25,50):", np.round(np.quantile(im.pd, [0, .01, .05, .25, .5]), 3))
print(f"events per image: mean {im.events.mean():.0f}, median {im.events.median():.0f}, p95 {im.events.quantile(.95):.0f}, max {im.events.max()}")
miss = S[~S.delivered]
print("\nmissed ships by size / contrast / scene / border:")
print(pd.cut(miss.areaMask, [0, 25, 50, 100, 250, 1e9]).value_counts().sort_index().to_string())
print("scene:", miss.scene.value_counts().to_dict(), "| near border (<8px):", int((miss.distBorder < 8).sum()), "| dxMax < 1.0:", int((miss.dxMax < 1.0).sum()))
im.to_csv(os.path.join(H.RES, "pd_study", "pooldetA_per_image_pd.csv"), index=False)
miss[["name", "img", "gt", "areaMask", "bw", "bh", "dxMax", "distBorder", "scene", "split"]].to_csv(os.path.join(H.RES, "pd_study", "pooldetA_missed_ships.csv"), index=False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(im.pd, bins=np.linspace(0, 1.0001, 41), color="#4292c6"); ax[0].set_yscale("log"); ax[0].set_xlabel("per-image prescreen Pd"); ax[0].set_ylabel("images (log)")
ax[0].set_title(f"{len(im)} images: {100*(im.pd==1).mean():.1f} % have every ship delivered")
ax[1].scatter(im.events, im.pd + np.random.RandomState(0).uniform(-.004, .004, len(im)), s=4, alpha=.3); ax[1].set_xscale("log"); ax[1].set_xlabel("events per image"); ax[1].set_ylabel("Pd (jittered)")
fig.tight_layout(); fig.savefig(os.path.join(H.RES, "pd_study", "fig_pooldetA_fullsweep.png"), dpi=140)
json.dump(dict(pd_all=float(S.delivered.mean()), n_ships=int(len(S)), missed=int((~S.delivered).sum())), open(os.path.join(H.RES, "pd_study", "pooldetA_fullsweep.json"), "w"))
