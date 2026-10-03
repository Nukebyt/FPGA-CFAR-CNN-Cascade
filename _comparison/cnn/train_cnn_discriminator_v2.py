# -*- coding: utf-8 -*-
"""Improved Paper 2 CNN discriminator training, targeting higher PRECISION
specifically (not just overall accuracy, which is an easy number here given
class imbalance -- rejecting every candidate already scores 84.65% accuracy
on this dataset, so accuracy alone does not demonstrate the discriminator is
doing useful work; precision -- "when the cascade says ship, how often is it
right" -- is the metric that actually matters for a deployed system and the
one this revision targets).

Changes from train_cnn_discriminator.py, each for a stated reason:
  1. Data augmentation (horizontal/vertical flip, 90-degree rotation) -- SAR
     ship orientation relative to the sensor track is arbitrary, so these
     are label-preserving transforms that multiply effective training data
     at zero extra collection cost.
  2. Wider network (16/32 conv filters instead of 8/16, FC1 64 instead of
     32) with BatchNorm after each conv -- the original model is small
     enough that under-capacity, not over-capacity, is the more likely
     ceiling; DSP/ALM budget on the DE10 target has ample room left (~88%
     of 112 DSP blocks, ~58% of 41,910 ALMs free after Weibull).
  3. Longer training (60 epochs) with cosine LR annealing and early
     stopping on validation PRECISION AT A FIXED MINIMUM RECALL (not raw
     F1 or accuracy), so model selection is aligned with the actual target.
  4. A post-training precision/recall operating-point sweep, reported in
     full rather than silently picking whichever threshold hits a target
     number -- moving the decision threshold trades recall for precision
     mechanically, and that tradeoff is reported explicitly here so a
     91%-precision claim can be read alongside the recall it costs.
"""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

MAT_PATH = r"F:\Projects\CFAR\_comparison\Results\cnn_patches.mat"
OUT_DIR = r"F:\Projects\CFAR\_comparison\Results"

torch.manual_seed(20260925)
np.random.seed(20260925)


def load_patches(path):
    with h5py.File(path, 'r') as f:
        patches = np.array(f['patches'])
        labels = np.array(f['labels']).squeeze().astype(bool)
        img_refs = f['imgName'][:].squeeze()
        img_names = []
        for ref in img_refs:
            obj = f[ref]
            s = ''.join(chr(c) for c in obj[:].squeeze())
            img_names.append(s)
        img_names = np.array(img_names)
    return patches.astype(np.float32), labels, img_names


class PatchDataset(Dataset):
    def __init__(self, patches, labels, mean, std, augment=False):
        self.patches = (patches - mean) / std
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
        y = torch.tensor(self.labels[i])
        return x, y


class ShipDiscriminatorV2(nn.Module):
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
    patches, labels, img_names = load_patches(MAT_PATH)
    n = len(labels)
    print(f"Loaded {n} patches ({labels.sum()} positive, {(~labels).sum()} negative)")

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
    print(f"Train-set normalization: mean={mean:.4f}, std={std:.4f}")

    train_ds = PatchDataset(patches[train_mask], labels[train_mask], mean, std, augment=True)
    val_ds = PatchDataset(patches[val_mask], labels[val_mask], mean, std, augment=False)
    test_ds = PatchDataset(patches[test_mask], labels[test_mask], mean, std, augment=False)

    train_labels = labels[train_mask]
    n_pos, n_neg = train_labels.sum(), (~train_labels).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test_ds, batch_size=512)

    model = ShipDiscriminatorV2()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")

    n_epochs = 60
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    MIN_RECALL = 0.80  # model-selection constraint: don't let precision optimization collapse recall
    best_score = -1
    best_state = None
    best_epoch = -1

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits, val_y = [], []
            for xb, yb in val_dl:
                val_logits.append(model(xb))
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
        for xb, yb in test_dl:
            test_logits.append(model(xb))
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
        'test_metrics_thresh0.5': m05,
        'sweep': sweep_results,
        'best_epoch': best_epoch,
    }, OUT_DIR + r"\cnn_discriminator_v2.pt")
    print("\nSaved", OUT_DIR + r"\cnn_discriminator_v2.pt")


if __name__ == '__main__':
    main()
