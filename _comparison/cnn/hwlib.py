# -*- coding: utf-8 -*-
"""Shared library for the hardware-constrained Paper 2 CNN study (v4+).

Data   : Results/cnn_patches_hw.mat  (extract_cnn_patches_hw.m) -- Weibull
         prescreen at the DEPLOYED window (sli=17/guard=13), Pfa=1e-3, EVERY
         cluster kept (true ~1.7% prevalence), uint8 40x40 log-amplitude.
Metric : the cascade is judged per SHIP and per IMAGE, not per cluster:
           ship retention   = fraction of CFAR-reachable ships with >=1
                              accepted ship-overlapping cluster
           FA / image       = accepted false-alarm clusters per image
         (CFAR alone passes ~1,670 FA/image at 100% retention.)
"""
import os
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

RES = r"F:\Projects\CFAR\_comparison\Results"
VARIANT = os.environ.get("HWDATA", "40")          # "40": full prevalence 40x40 | "64": gated 64x64
_FILES = {"40": ("cnn_patches_hw.mat", "hw_cache"), "64": ("cnn_patches_hw64.mat", "hw_cache64"),
          "full32": ("cnn_patches_hw_full32.mat", "hw_cache_full32"),
          "hwspec": ("cnn_patches_hwspec.mat", "hw_cache_hwspec")}   # hardware-faithful: NMS trigger, pooled store, gate = x - c1   # full HRSID, gated, stored already 2x2-pooled (32x32)
if VARIANT.startswith("pooldet"):                 # pooled-domain prescreen datasets (pd_study/extract_cnn_patches_pooldet.m), several part files
    _FILES[VARIANT] = (f"cnn_patches_{VARIANT}_p*.mat", f"hw_cache_{VARIANT}")
MAT = os.path.join(RES, _FILES[VARIANT][0])
CACHE = os.path.join(RES, _FILES[VARIANT][1])
GATE_TAU = float(os.environ["HWGATE"]) if "HWGATE" in os.environ else None
NEG_FLOOR = -1e9                                  # score given to gate-rejected clusters
XLO, XHI = -0.40, 2.80
SPLIT_SEED = int(os.environ.get("HWSPLIT", 20260925))


# --------------------------------------------------------------------- data
def _build_cache():
    os.makedirs(CACHE, exist_ok=True)
    if VARIANT.startswith("pooldet"):
        import glob
        parts = sorted(glob.glob(MAT))
        pat, per, cxy, imgn = [], {k: [] for k in ("labels", "imgIdx", "gtIdx", "gtIdx2", "compArea", "xt", "c1", "gate")}, [], []
        ns = nt = None
        for fn in parts:
            with h5py.File(fn, "r") as f:
                pat.append(np.array(f["patches"]))
                for k in per:
                    per[k].append(np.array(f[k]).reshape(-1))
                cxy.append(np.array(f["tyx"]).reshape(2, -1))
                a, b = np.array(f["imgNShips"]).reshape(-1), np.array(f["imgNTrig"]).reshape(-1)
                ns = a if ns is None else np.maximum(ns, a); nt = b if nt is None else np.maximum(nt, b)
        np.save(os.path.join(CACHE, "patches.npy"), np.concatenate(pat, axis=0))
        g = {k: np.concatenate(v) for k, v in per.items()}
        n = len(g["labels"]); z = np.zeros(n, np.float32)
        np.savez(os.path.join(CACHE, "meta.npz"), gtIdx2=g["gtIdx2"], labels=g["labels"], imgIdx=g["imgIdx"], gtIdx=g["gtIdx"], area=g["compArea"], bboxH=z, bboxW=z,
                 peakX=g["xt"], meanX=g["xt"], c1=g["c1"], c2=z, imgNShips=ns, cxy=np.concatenate(cxy, axis=1), gate3=g["gate"], imgNTrig=nt)
        return
    if VARIANT == "hwspec":
        with h5py.File(MAT, "r") as f:
            np.save(os.path.join(CACHE, "patches.npy"), np.array(f["patches"]))
            g = lambda k: np.array(f[k]).squeeze()
            n = len(g("labels"))
            z = np.zeros(n, np.float32)
            np.savez(os.path.join(CACHE, "meta.npz"), labels=g("labels"), imgIdx=g("imgIdx"), gtIdx=g("gtIdx"),
                     area=g("compArea"), bboxH=z, bboxW=z, peakX=g("xt"), meanX=g("xt"), c1=g("c1"), c2=z,
                     imgNShips=g("imgNShips"), cxy=np.array(f["tyx"]), gate3=g("gate"), imgNTrig=g("imgNTrig"))
        return
    with h5py.File(MAT, "r") as f:
        np.save(os.path.join(CACHE, "patches.npy"), np.array(f["patches"]))
        d = {k: np.array(f[k]).squeeze() for k in
             ["labels", "imgIdx", "gtIdx", "area", "bboxH", "bboxW",
              "peakX", "meanX", "c1", "c2", "imgNShips"]}
        d["cxy"] = np.array(f["cxy"])
        if "gate3" in f:
            d["gate3"] = np.array(f["gate3"]).squeeze()
        np.savez(os.path.join(CACHE, "meta.npz"), **d)


class HWData:
    def __init__(self):
        if not os.path.exists(os.path.join(CACHE, "patches.npy")):
            _build_cache()
        self.patches = np.load(os.path.join(CACHE, "patches.npy"), mmap_mode="r")
        m = np.load(os.path.join(CACHE, "meta.npz"))
        self.labels = m["labels"].astype(bool)
        self.img = m["imgIdx"].astype(np.int64) - 1
        self.gt = m["gtIdx"].astype(np.int64)
        self.area = m["area"].astype(np.float32)
        self.peakX = m["peakX"].astype(np.float32)
        self.meanX = m["meanX"].astype(np.float32)
        self.c1 = m["c1"].astype(np.float32)
        self.c2 = m["c2"].astype(np.float32)
        self.nships = m["imgNShips"].astype(np.int64)
        self.n = len(self.labels)
        self.P = self.patches.shape[1]
        gp = os.path.join(CACHE, "gate3x3.npy")
        if "gate3" in m.files and not os.path.exists(gp):
            np.save(gp, m["gate3"].astype(np.float32))        # computed in MATLAB at full resolution
        if not os.path.exists(gp):
            o = (self.P - 3) // 2
            sc = (XHI - XLO) / 255.0
            g = np.empty(self.n, np.float32)
            for i in range(0, self.n, 50000):
                blk = np.asarray(self.patches[i:i + 50000, o:o + 3, o:o + 3], dtype=np.float32)
                g[i:i + 50000] = blk.mean(axis=(1, 2)) * sc + XLO
            np.save(gp, g - self.c1)
        self.gate = np.load(gp)
        n_img = len(self.nships)
        order = np.arange(n_img)
        np.random.RandomState(SPLIT_SEED).shuffle(order)
        ntr, nva = int(0.7 * n_img), int(0.15 * n_img)
        self.split_imgs = {"train": order[:ntr], "val": order[ntr:ntr + nva],
                           "test": order[ntr + nva:]}
        self.idx = {}      # every cluster of the split's images (evaluation universe)
        self.pidx = {}     # clusters that actually reach the CNN (pass the gate, if any)
        for k, imgs in self.split_imgs.items():
            self.idx[k] = np.where(np.isin(self.img, imgs))[0]
            ok = (self.gate[self.idx[k]] >= GATE_TAU) if GATE_TAU is not None else np.ones(len(self.idx[k]), bool)
            self.pidx[k] = self.idx[k][ok]
        # CFAR-only candidate count per image (true prevalence), for reduction factors
        if VARIANT == "40":
            self.cand_per_img = {k: len(self.idx[k]) / len(v) for k, v in self.split_imgs.items()}
        elif VARIANT == "hwspec" or VARIANT.startswith("pooldet"):
            ntrig = m["imgNTrig"].astype(np.float64)
            self.cand_per_img = {k: float(ntrig[v].mean()) for k, v in self.split_imgs.items()}
        elif VARIANT == "full32":
            from scipy.io import loadmat
            cnt = loadmat(os.path.join(RES, "cfar_cluster_counts.mat"))["cnt"].squeeze().astype(np.float64)
            self.cand_per_img = {k: float(cnt[v].mean()) for k, v in self.split_imgs.items()}
        else:
            full = np.load(os.path.join(RES, "hw_cache", "meta.npz"))["imgIdx"].astype(np.int64) - 1
            cnt = np.bincount(full, minlength=len(self.nships))
            self.cand_per_img = {k: float(cnt[v].mean()) for k, v in self.split_imgs.items()}

    def tensors(self):
        """CPU torch views (uint8 patches stay in RAM)."""
        return (torch.from_numpy(np.ascontiguousarray(self.patches)),
                torch.from_numpy(self.c1))


# --------------------------------------------------------------- preprocess
MEAN, STD = 1.4674, 0.4974   # global log-amplitude stats (v1-v3 train set; same sensor)


def prep(xq, c1, norm, size, jitter=0, train=False, down=1):
    P = xq.shape[1]
    """xq uint8 (B,40,40) on device -> float (B,1,size,size) network input.

    norm: 'global' (fixed mean/std), 'pmean' (subtract the patch's own mean --
          one accumulator + subtractor in hardware), 'c1' (subtract the
          prescreen's local log-mean at the centroid, free from Weibull's own
          moment engine).
    """
    B = xq.shape[0]
    o = (P - size) // 2
    if train and jitter > 0:
        dy = int(np.random.randint(-jitter, jitter + 1))
        dx = int(np.random.randint(-jitter, jitter + 1))
    else:
        dy = dx = 0
    x = xq[:, o + dy:o + dy + size, o + dx:o + dx + size].float()
    x = x * ((XHI - XLO) / 255.0) + XLO
    if norm == "global":
        x = (x - MEAN) / STD
    elif norm == "pmean":
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / STD
    elif norm == "c1":
        x = (x - c1.view(B, 1, 1)) / STD
    else:
        raise ValueError(norm)
    if train:  # dihedral augmentation (label-preserving: ship heading is arbitrary)
        m = torch.rand(B, 3, device=x.device) < 0.5
        x = torch.where(m[:, 0].view(B, 1, 1), x.flip(2), x)
        x = torch.where(m[:, 1].view(B, 1, 1), x.flip(1), x)
        x = torch.where(m[:, 2].view(B, 1, 1), x.transpose(1, 2), x)
    x = x.unsqueeze(1)
    if down > 1:
        x = F.avg_pool2d(x, down)
    return x


def prep_cfg(xq, c1, cfg, jitter=0, train=False, shift=None):
    """Network input(s) for a config: a tensor (single tower) or a tuple
    (centre tower, context tower).  The dihedral augmentation is applied to the
    whole patch BEFORE cropping so both towers see the same transform."""
    B, P = xq.shape[0], xq.shape[1]
    x = xq.float() * ((XHI - XLO) / 255.0) + XLO
    norm = cfg["norm"]
    if norm == "global":
        x = (x - MEAN) / STD
    elif norm == "pmean":
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / STD
    else:
        x = (x - c1.view(B, 1, 1)) / STD
    if train:
        m = torch.rand(B, 3, device=x.device) < 0.5
        x = torch.where(m[:, 0].view(B, 1, 1), x.flip(2), x)
        x = torch.where(m[:, 1].view(B, 1, 1), x.flip(1), x)
        x = torch.where(m[:, 2].view(B, 1, 1), x.transpose(1, 2), x)
    if shift is not None:
        dy, dx = shift
    elif train and jitter > 0:
        dy = int(np.random.randint(-jitter, jitter + 1)); dx = int(np.random.randint(-jitter, jitter + 1))
    else:
        dy = dx = 0
    PAD = 0
    if dy or dx:                       # a shifted window can reach past the stored patch: replicate the edge
        PAD = max(abs(dy), abs(dx))
        x = F.pad(x.unsqueeze(1), (PAD, PAD, PAD, PAD), mode="replicate").squeeze(1)

    def crop(size, down):
        o = (P - size) // 2 + PAD
        y0 = o + dy; x0 = o + dx
        t = x[:, y0:y0 + size, x0:x0 + size].unsqueeze(1)
        return F.avg_pool2d(t, down) if down > 1 else t

    xa = crop(cfg["size"], cfg.get("down", 1))
    if cfg.get("convs2"):
        return (xa, crop(cfg["size2"], cfg.get("down2", 1)))
    return xa


# -------------------------------------------------------------------- model
def parse_convs(s):
    """'5-8-1,5-16-1' -> [(k, cout, pool_after), ...]"""
    out = []
    for t in s.split(","):
        k, c, p = t.split("-")
        out.append((int(k), int(c), int(p)))
    return out


class Net(nn.Module):
    """Hardware-mappable CNN: valid convs (no padding logic), BatchNorm (folded
    into the conv at export), ReLU, 2x2 maxpool, flatten, FC stack."""

    def __init__(self, size, convs, fcs, drop=0.3):
        super().__init__()
        layers, c, s = [], 1, size
        self.macs, self.params = 0, 0
        for (k, co, pool) in convs:
            layers += [nn.Conv2d(c, co, k, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            s = s - k + 1
            self.macs += s * s * co * c * k * k
            self.params += co * c * k * k + co          # BN folds to a bias
            c = co
            if pool:
                layers.append(nn.MaxPool2d(2))
                s //= 2
        self.features = nn.Sequential(*layers)
        flat = c * s * s
        self.flat = flat
        fl = []
        d = flat
        for h in fcs:
            fl += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            self.macs += d * h
            self.params += d * h + h
            d = h
        if drop > 0 and fcs:
            fl.insert(len(fl) - 0, nn.Dropout(drop))
        fl.append(nn.Linear(d, 1))
        self.macs += d
        self.params += d + 1
        self.classifier = nn.Sequential(*fl)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1)).squeeze(1)


class Net2(nn.Module):
    """Centre tower (full-res crop) + context tower (wide crop, down-sampled);
    flattened features are concatenated before the FC stack.  Hardware: the two
    towers run back-to-back on the same MAC array."""

    def __init__(self, cfg):
        super().__init__()
        ta = Net(cfg["size"] // cfg.get("down", 1), parse_convs(cfg["convs"]), [], 0.0)
        tb = Net(cfg["size2"] // cfg.get("down2", 1), parse_convs(cfg["convs2"]), [], 0.0)
        self.fa, self.fb = ta.features, tb.features
        flat = ta.flat + tb.flat
        self.flat = flat
        self.macs = ta.macs - 1 + tb.macs - 1          # the dummy 1-wide heads are not part of this net
        self.params = ta.params - (ta.flat + 1) + tb.params - (tb.flat + 1)
        fcs = [int(v) for v in cfg["fcs"].split(",")] if cfg["fcs"] else []
        fl, d = [], flat
        for h in fcs:
            fl += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            self.macs += d * h; self.params += d * h + h; d = h
        if cfg.get("drop", 0.3) > 0 and fcs:
            fl.append(nn.Dropout(cfg.get("drop", 0.3)))
        fl.append(nn.Linear(d, 1)); self.macs += d; self.params += d + 1
        self.classifier = nn.Sequential(*fl)

    def forward(self, x):
        xa, xb = x
        return self.classifier(torch.cat([self.fa(xa).flatten(1), self.fb(xb).flatten(1)], 1)).squeeze(1)


def build(cfg):
    if cfg.get("convs2"):
        return Net2(cfg)
    return Net(cfg["size"] // cfg.get("down", 1), parse_convs(cfg["convs"]),
               [int(v) for v in cfg["fcs"].split(",")] if cfg["fcs"] else [],
               cfg.get("drop", 0.3))


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def score(model, data_t, c1_t, idx, cfg, dev, chunk=8192, shift=None):
    model.eval()
    out = []
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        xq = data_t[j].to(dev, non_blocking=True)
        c1 = c1_t[j].to(dev)
        x = prep_cfg(xq, c1, cfg, shift=shift)
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def full_scores(model, d, split, data_t, c1_t, cfg, dev, shift=None):
    """Scores aligned with d.idx[split]; gate-rejected clusters get NEG_FLOOR."""
    idx = d.idx[split]
    out = np.full(len(idx), NEG_FLOOR, dtype=np.float64)
    ok = np.isin(idx, d.pidx[split]) if GATE_TAU is not None else np.ones(len(idx), bool)
    out[ok] = score(model, data_t, c1_t, idx[ok], cfg, dev, shift=shift)
    return out


# ------------------------------------------------------------------ metrics
RETS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def ship_table(d, idx):
    """Per-ship grouping for the cluster subset idx (positive clusters only)."""
    pos = d.labels[idx]
    key = d.img[idx][pos] * 256 + d.gt[idx][pos]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uniq, start = np.unique(ks, return_index=True)
    return pos, order, start


def cascade_curve(d, idx, scores, rets=RETS):
    """Threshold-free comparison: FA/image at fixed ship retention (retention is
    relative to the CFAR-reachable ships, i.e. those with >=1 positive cluster)."""
    pos, order, start = ship_table(d, idx)
    n_img = len(np.unique(d.img[idx]))
    sp = scores[pos][order]
    ship_max = np.maximum.reduceat(sp, start)
    sm = np.sort(ship_max)
    neg = np.sort(scores[~pos])
    out = {}
    for r in rets:
        k = int(np.floor((1 - r) * len(sm)))
        t = sm[k]
        fa = len(neg) - np.searchsorted(neg, t, side="left")
        acc_pos = int((scores[pos] >= t).sum())
        out[str(r)] = dict(thr=float(t), fa_per_img=fa / n_img,
                           cand_per_img=(fa + acc_pos) / n_img)
    return out


def split_name_of(d, idx):
    for k in d.idx:
        if len(d.idx[k]) == len(idx) and d.idx[k][0] == idx[0]:
            return k
    return None


def at_threshold(d, idx, scores, thr):
    """Deployment-style numbers at a fixed threshold (chosen on validation)."""
    pos, order, start = ship_table(d, idx)
    n_img = len(np.unique(d.img[idx]))
    sp = scores[pos][order]
    ship_max = np.maximum.reduceat(sp, start)
    neg = scores[~pos]
    acc = scores >= thr
    tp, fp = int((acc & d.labels[idx]).sum()), int((acc & ~d.labels[idx]).sum())
    return dict(thr=float(thr),
                ship_retention=float((ship_max >= thr).mean()),
                fa_per_img=fp / n_img,
                cand_per_img=(tp + fp) / n_img,
                cluster_precision=tp / max(tp + fp, 1),
                cluster_recall=tp / max(int(pos.sum()), 1),
                reduction=(d.cand_per_img.get(split_name_of(d, idx), len(idx) / n_img) * n_img) / max(tp + fp, 1))


def thr_for_retention(d, idx, scores, r):
    pos, order, start = ship_table(d, idx)
    sm = np.sort(np.maximum.reduceat(scores[pos][order], start))
    return float(sm[int(np.floor((1 - r) * len(sm)))])


def ap_score(d, idx, scores):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(d.labels[idx], scores))


def baseline_scores(d, idx):
    """Trivial hardware-free-lunch gates, same metric, for context."""
    return {"peakX": d.peakX[idx], "peak-c1": d.peakX[idx] - d.c1[idx],
            "area": d.area[idx], "meanX-c1": d.meanX[idx] - d.c1[idx]}
