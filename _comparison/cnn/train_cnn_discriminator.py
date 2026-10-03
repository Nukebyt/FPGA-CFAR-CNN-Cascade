# -*- coding: utf-8 -*-
"""Train the Paper 2 CNN discriminator (real-ship vs. false-alarm) on the
Weibull-CFAR-flagged patches extract_cnn_patches.m produced.

Architecture is deliberately small and hardware-shaped (see the docstring
below for the DE10/Cyclone V resource reasoning) -- this is NOT a general
SAR-ATR classifier, it only ever sees patches the CFAR prescreen already
flagged, so the discrimination task (real ship vs. clutter false alarm) is
much easier than open-set detection.

    Conv1: 8  filters, 5x5, stride 1  -> 28x28x8,  ReLU, 2x2 maxpool -> 14x14x8
    Conv2: 16 filters, 5x5, stride 1  -> 10x10x16, ReLU, 2x2 maxpool -> 5x5x16
    FC1:   400 -> 32, ReLU
    FC2:   32 -> 1, sigmoid

    ~16.3k parameters total (Conv1 208, Conv2 3216, FC1 12832, FC2 33) --
    at INT8 that is ~16KB, a trivial fraction of the Cyclone V
    5CSXFC6D6F31C6's 5.53 Mbit on-chip memory, and comfortably less than
    the ~5.35 Mbit left after Weibull's own front end (which uses 309,183
    bits, 5%, per ROADMAP.md's resource probe).
    ~490k MACs per patch classification -- at even 16 DSP-based MAC units
    running in parallel at Weibull's existing 50 MHz clock, one patch
    classifies in ~31k cycles (~0.6ms), fast relative to how often a
    cluster actually needs classifying (a sparse, per-detection event, not
    a per-pixel one).

Train/val/test is split by SOURCE IMAGE, never by patch -- patches from the
same image share clutter statistics and would leak across a patch-level
split.
"""
import sys
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
        patches = np.array(f['patches'])          # (N, P, P) after h5py's axis reversal of MATLAB's (P,P,N)
        labels = np.array(f['labels']).squeeze().astype(bool)
        # imgName is a MATLAB cellstr -> array of HDF5 object references
        img_refs = f['imgName'][:].squeeze()
        img_names = []
        for ref in img_refs:
            obj = f[ref]
            s = ''.join(chr(c) for c in obj[:].squeeze())
            img_names.append(s)
        img_names = np.array(img_names)
    return patches.astype(np.float32), labels, img_names


class PatchDataset(Dataset):
    def __init__(self, patches, labels, mean, std):
        self.patches = (patches - mean) / std
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.patches[i]).unsqueeze(0)  # 1xPxP
        y = torch.tensor(self.labels[i])
        return x, y


class ShipDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(5 * 5 * 16, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)  # logits


def main():
    print("Loading", MAT_PATH)
    patches, labels, img_names = load_patches(MAT_PATH)
    n = len(labels)
    print(f"Loaded {n} patches ({labels.sum()} positive, {(~labels).sum()} negative), patch shape {patches.shape[1:]}")

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
    print(f"Image-disjoint split: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test images")
    print(f"Patch counts: {train_mask.sum()} train / {val_mask.sum()} val / {test_mask.sum()} test")

    mean = patches[train_mask].mean()
    std = patches[train_mask].std()
    print(f"Train-set normalization: mean={mean:.4f}, std={std:.4f}")

    train_ds = PatchDataset(patches[train_mask], labels[train_mask], mean, std)
    val_ds = PatchDataset(patches[val_mask], labels[val_mask], mean, std)
    test_ds = PatchDataset(patches[test_mask], labels[test_mask], mean, std)

    train_labels = labels[train_mask]
    n_pos, n_neg = train_labels.sum(), (~train_labels).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"pos_weight for BCE (class balancing): {pos_weight.item():.3f}")

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)
    test_dl = DataLoader(test_ds, batch_size=256)

    model = ShipDiscriminator()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1
    best_state = None
    for epoch in range(30):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits, val_y = [], []
            for xb, yb in val_dl:
                val_logits.append(model(xb))
                val_y.append(yb)
            val_logits = torch.cat(val_logits)
            val_y = torch.cat(val_y)
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            tp = ((val_pred == 1) & (val_y == 1)).sum().item()
            fp = ((val_pred == 1) & (val_y == 0)).sum().item()
            fn = ((val_pred == 0) & (val_y == 1)).sum().item()
            tn = ((val_pred == 0) & (val_y == 0)).sum().item()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            acc = (tp + tn) / max(len(val_y), 1)
        print(f"epoch {epoch:2d}  val acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  (tp={tp} fp={fp} fn={fn} tn={tn})")
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits, test_y = [], []
        for xb, yb in test_dl:
            test_logits.append(model(xb))
            test_y.append(yb)
        test_logits = torch.cat(test_logits)
        test_y = torch.cat(test_y)
        test_pred = (torch.sigmoid(test_logits) > 0.5).float()
        tp = ((test_pred == 1) & (test_y == 1)).sum().item()
        fp = ((test_pred == 1) & (test_y == 0)).sum().item()
        fn = ((test_pred == 0) & (test_y == 1)).sum().item()
        tn = ((test_pred == 0) & (test_y == 0)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (tp + tn) / max(len(test_y), 1)
    print(f"\nHELD-OUT TEST (image-disjoint): acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  (tp={tp} fp={fp} fn={fn} tn={tn})")

    torch.save({
        'state_dict': model.state_dict(),
        'mean': mean, 'std': std,
        'test_metrics': {'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                          'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
    }, OUT_DIR + r"\cnn_discriminator.pt")
    print("Saved", OUT_DIR + r"\cnn_discriminator.pt")


if __name__ == '__main__':
    main()
