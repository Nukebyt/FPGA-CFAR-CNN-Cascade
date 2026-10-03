# -*- coding: utf-8 -*-
"""Reference points for the cascade metric: CFAR alone, and trivial
hardware-cheap gates (peak level, peak-over-local-mean, cluster area)."""
import json
import numpy as np
import hwlib as H

d = H.HWData()
out = {}
for split in ("val", "test"):
    idx = d.idx[split]
    n_img = len(d.split_imgs[split])
    pos = d.labels[idx]
    pos_t, order, start = H.ship_table(d, idx)
    n_reach = len(start)
    n_ships = int(d.nships[d.split_imgs[split]].sum())
    print(f"\n=== {split}: {n_img} images, {len(idx)} clusters ({pos.sum()} ship-overlapping, "
          f"{100*pos.mean():.2f}%), {n_ships} GT ships, {n_reach} CFAR-reachable "
          f"({100*n_reach/n_ships:.1f}%)")
    print(f"CFAR alone: {(~pos).sum()/n_img:.1f} FA/img, {len(idx)/n_img:.1f} candidates/img at 100% retention")
    out[split] = dict(n_img=n_img, clusters=len(idx), pos=int(pos.sum()), ships=n_ships,
                      reachable=n_reach, cfar_fa_per_img=float((~pos).sum() / n_img), gates={})
    for name, s in H.baseline_scores(d, idx).items():
        c = H.cascade_curve(d, idx, s.astype(np.float64))
        out[split]["gates"][name] = c
        print(f"  gate {name:9s} FA/img @ret 80/85/90/95: " +
              " ".join(f"{c[str(r)]['fa_per_img']:7.1f}" for r in H.RETS) +
              f"   AP={H.ap_score(d, idx, s):.4f}")
json.dump(out, open(H.RES + r"\hw\baselines.json", "w"), indent=1)
