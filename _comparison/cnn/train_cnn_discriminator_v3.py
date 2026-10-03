# -*- coding: utf-8 -*-
"""Paper 2 CNN discriminator, v3: hard-negative mining + CFAR-statistics
feature fusion, on top of v2's architecture (BatchNorm/Dropout/augmentation).

WHY THESE TWO CHANGES SPECIFICALLY
-----------------------------------
v1 -> v2 (wider net, BatchNorm, augmentation, longer training) barely moved
the default-threshold operating point (v1: P=0.6830/R=0.8706, v2: P=0.6797/
R=0.8926 at thresh=0.5) -- the network's decision boundary was already close
to what that architecture/data combination can produce. Pushing precision
further by raising the threshold trades it directly against recall (v2 at
thresh=0.90: P=0.9035/R=0.6377). That plateau means the ceiling is set by
what the model is SHOWN, not by its capacity:

  1. HARD-NEGATIVE MINING. The training set's negatives are a random 8:1
     subsample per image (extract_cnn_patches.m) -- mostly flat sea, which
     the network already separates from ships almost perfectly and gets
     near-zero learning signal from. The false positives that actually cost
     precision are the CONFUSABLE ones: wave streaks, azimuth ghosts,
     coastline glare. v2's own checkpoint is used here to score every
     training negative; the top 25% by predicted ship-probability (the ones
     v2 itself is least sure about) are oversampled 3x via a
     WeightedRandomSampler, concentrating training signal on the boundary
     cases instead of examples already solved.

  2. CFAR-STATISTICS FEATURE FUSION. The CNN sees only a raw 32x32 pixel
     patch -- it has to re-derive contrast, texture, and footprint shape
     from pixels alone, on a network too small for much redundant
     computation. cfar_front_end/cluster geometry already carry this
     information cheaply. The original extraction (extract_cnn_patches.m)
     only kept clusterArea, not per-cluster contrast/shape/texture, and
     re-running that extraction now would contend with the multi-hour
     comparison sweep this session is also running -- so the remaining
     features (contrast, local texture, footprint aspect ratio) are instead
     derived directly from the already-saved 32x32 patch pixels themselves
     (see compute_patch_features below). This is an approximation of the
     true connected-component footprint (which was collapsed to a centroid+
     area at extraction time and is not recoverable from the patch alone),
     not a re-derivation of it -- stated explicitly, not hidden, since it
     matters for how strongly to trust the aspect-ratio/footprint-area
     features specifically vs. the pixel-domain ones (contrast/std/skew),
     which are exact.

Both changes reuse v2's train/val/test image split (same seed) so results
are directly comparable.
"""
import os
import h5py
import numpy as np
from scipy import ndimage
from scipy.stats import skew as scipy_skew
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

MAT_PATH = r"F:\Projects\CFAR\_comparison\Results\cnn_patches.mat"
V2_CKPT = r"F:\Projects\CFAR\_comparison\Results\cnn_discriminator_v2.pt"
OUT_DIR = r"F:\Projects\CFAR\_comparison\Results"

torch.manual_seed(20260928)
np.random.seed(20260928)

N_FEAT = 8  # [clusterArea, peak, background, contrast, std, skew, aspectRatio, footprintArea]


def load_patches(path):
    with h5py.File(path, 'r') as f:
        patches = np.array(f['patches'])
        labels = np.array(f['labels']).squeeze().astype(bool)
        cluster_area = np.array(f['clusterArea']).squeeze().astype(np.float32)
        img_refs = f['imgName'][:].squeeze()
        img_names = []
        for ref in img_refs:
            obj = f[ref]
            s = ''.join(chr(c) for c in obj[:].squeeze())
            img_names.append(s)
        img_names = np.array(img_names)
    return patches.astype(np.float32), labels, img_names, cluster_area


def compute_patch_features(patches, cluster_area):
    """Per-patch engineered features, derived directly from the saved 32x32
    log-amplitude crop (fe.x domain) plus the cluster's true pixel-count
    area saved at extraction time. See module docstring for what is exact
    (pixel-statistics) vs. approximate (footprint shape, re-derived from the
    patch itself since the true connected-component mask was not kept)."""
    n = patches.shape[0]
    feat = np.zeros((n, N_FEAT), dtype=np.float32)
    cy, cx = patches.shape[1] // 2, patches.shape[2] // 2
    for i in range(n):
        p = patches[i]
        peak = float(p.max())
        bg = float(np.percentile(p, 20))
        contrast = peak - bg
        std = float(p.std())
        sk = float(scipy_skew(p.ravel()))

        thr = bg + 0.5 * contrast
        mask = p > thr
        lbl, _ = ndimage.label(mask)
        cid = lbl[cy, cx]
        if cid == 0:
            aspect, foot_area = 1.0, 1.0
        else:
            ys, xs = np.where(lbl == cid)
            fh = ys.max() - ys.min() + 1
            fw = xs.max() - xs.min() + 1
            aspect = max(fw, fh) / max(min(fw, fh), 1)
            foot_area = float(len(ys))

        feat[i] = [cluster_area[i], peak, bg, contrast, std, sk, aspect, foot_area]
    return feat


class PatchDataset(Dataset):
    def __init__(self, patches, feat, labels, mean, std, feat_mean, feat_std, augment=False):
        self.patches = (patches - mean) / std
        self.feat = (feat - feat_mean) / feat_std
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.patches[i]
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[::-1, :]
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k)
            x = np.ascontiguousarray(x)
        x = torch.from_numpy(x).unsqueeze(0)
        f = torch.from_numpy(self.feat[i])
        y = torch.tensor(self.labels[i])
        return x, f, y


class ShipDiscriminatorV2(nn.Module):
    """Kept identical to train_cnn_discriminator_v2.py so its checkpoint
    loads unmodified for hard-negative scoring."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(5 * 5 * 32, 64)
        self.fc2 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(1)


class ShipDiscriminatorV3(nn.Module):
    """Same conv backbone as v2, plus a small feature-fusion head."""
    def __init__(self, n_feat=N_FEAT):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc_img = nn.Linear(5 * 5 * 32, 64)
        self.fc_feat = nn.Linear(n_feat, 16)
        self.fc_fusion = nn.Linear(64 + 16, 32)
        self.fc_out = nn.Linear(32, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, feat):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.flatten(1)
        img_emb = F.relu(self.fc_img(x))
        feat_emb = F.relu(self.fc_feat(feat))
        h = torch.cat([img_emb, feat_emb], dim=1)
        h = self.drop(F.relu(self.fc_fusion(h)))
        return self.fc_out(h).squeeze(1)


def compute_metrics(logits, y, thresh=0.5):
    pred = (torch.sigmoid(logits) > thresh).float()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(len(y), 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, accuracy=acc)


def main():
    print("Loading", MAT_PATH)
    patches, labels, img_names, cluster_area = load_patches(MAT_PATH)
    n = len(labels)
    print(f"Loaded {n} patches ({labels.sum()} positive, {(~labels).sum()} negative)")

    print("Computing per-patch engineered features (contrast/texture/footprint)...")
    feat = compute_patch_features(patches, cluster_area)

    # Identical split logic/seed to v2 for a fair before/after comparison.
    unique_imgs = np.unique(img_names)
    rng = np.random.RandomState(20260925)
    rng.shuffle(unique_imgs)
    n_img = len(unique_imgs)
    n_train = int(0.7 * n_img)
    n_val = int(0.15 * n_img)
    train_imgs = set(unique_imgs[:n_train])
    val_imgs = set(unique_imgs[n_train:n_train + n_val])
    test_imgs = set(unique_imgs[n_train + n_val:])

    train_mask = np.array([im in train_imgs for im in img_names])
    val_mask = np.array([im in val_imgs for im in img_names])
    test_mask = np.array([im in test_imgs for im in img_names])

    mean = patches[train_mask].mean()
    std = patches[train_mask].std()
    feat_mean = feat[train_mask].mean(axis=0, keepdims=True)
    feat_std = feat[train_mask].std(axis=0, keepdims=True) + 1e-6
    print(f"Train-set patch normalization: mean={mean:.4f}, std={std:.4f}")
    print(f"Train-set feature normalization: mean={feat_mean.ravel()}, std={feat_std.ravel()}")

    train_ds = PatchDataset(patches[train_mask], feat[train_mask], labels[train_mask],
                             mean, std, feat_mean, feat_std, augment=True)
    val_ds = PatchDataset(patches[val_mask], feat[val_mask], labels[val_mask],
                           mean, std, feat_mean, feat_std, augment=False)
    test_ds = PatchDataset(patches[test_mask], feat[test_mask], labels[test_mask],
                            mean, std, feat_mean, feat_std, augment=False)

    train_labels = labels[train_mask]
    n_pos, n_neg = int(train_labels.sum()), int((~train_labels).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    # ---- Hard-negative mining: score every training negative with v2 -----
    print("\nLoading v2 checkpoint for hard-negative scoring:", V2_CKPT)
    v2_state = torch.load(V2_CKPT, weights_only=False)
    v2_model = ShipDiscriminatorV2()
    v2_model.load_state_dict(v2_state['state_dict'])
    v2_model.eval()

    train_patches_norm = (patches[train_mask] - mean) / std
    with torch.no_grad():
        scores = []
        bs = 1024
        for i in range(0, len(train_patches_norm), bs):
            xb = torch.from_numpy(train_patches_norm[i:i + bs]).unsqueeze(1)
            scores.append(torch.sigmoid(v2_model(xb)))
        scores = torch.cat(scores).numpy()

    neg_mask = ~train_labels
    neg_scores = scores[neg_mask]
    hard_cut = np.percentile(neg_scores, 75)  # top 25% by v2's own predicted ship-probability
    is_hard_neg = neg_mask & (scores >= hard_cut)
    print(f"Hard-negative mining: {is_hard_neg.sum()} / {neg_mask.sum()} training negatives "
          f"flagged hard (v2 score >= {hard_cut:.4f})")

    HARD_MULT = 3.0
    sample_weight = np.ones(len(train_labels), dtype=np.float64)
    sample_weight[is_hard_neg] = HARD_MULT
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(train_labels), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=128, sampler=sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test_ds, batch_size=512)

    model = ShipDiscriminatorV3()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params}")

    n_epochs = 60
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    MIN_RECALL = 0.80
    best_score = -1
    best_state = None
    best_epoch = -1

    for epoch in range(n_epochs):
        model.train()
        for xb, fb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb, fb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits, val_y = [], []
            for xb, fb, yb in val_dl:
                val_logits.append(model(xb, fb))
                val_y.append(yb)
            val_logits = torch.cat(val_logits)
            val_y = torch.cat(val_y)
        m = compute_metrics(val_logits, val_y, thresh=0.5)
        score = m['precision'] if m['recall'] >= MIN_RECALL else m['precision'] - 1.0
        print(f"epoch {epoch:2d}  lr={sched.get_last_lr()[0]:.5f}  val acc={m['accuracy']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}  score={score:.4f}")
        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    print(f"\nBest epoch: {best_epoch} (score={best_score:.4f})")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits, test_y = [], []
        for xb, fb, yb in test_dl:
            test_logits.append(model(xb, fb))
            test_y.append(yb)
        test_logits = torch.cat(test_logits)
        test_y = torch.cat(test_y)

    print("\n--- Held-out test set: precision/recall operating-point sweep ---")
    print(f"{'thresh':>7} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'tp':>6} {'fp':>6} {'fn':>6} {'tn':>6}")
    sweep_results = {}
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]:
        m = compute_metrics(test_logits, test_y, thresh=thresh)
        sweep_results[thresh] = m
        print(f"{thresh:7.2f} {m['accuracy']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['f1']:7.4f} "
              f"{m['tp']:6d} {m['fp']:6d} {m['fn']:6d} {m['tn']:6d}")

    m05 = compute_metrics(test_logits, test_y, thresh=0.5)
    torch.save({
        'state_dict': model.state_dict(),
        'mean': mean, 'std': std,
        'feat_mean': feat_mean, 'feat_std': feat_std,
        'test_metrics_thresh0.5': m05,
        'sweep': sweep_results,
        'best_epoch': best_epoch,
        'hard_neg_cut': hard_cut,
        'n_hard_neg': int(is_hard_neg.sum()),
    }, OUT_DIR + r"\cnn_discriminator_v3.pt")
    print("\nSaved", OUT_DIR + r"\cnn_discriminator_v3.pt")


if __name__ == '__main__':
    main()
