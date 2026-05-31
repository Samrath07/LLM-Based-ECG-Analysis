"""
dataset.py — Data loading, preprocessing, and DataLoader construction.

Expects:
    X_balanced_ecg.npy  — shape (N, 1000, 12)
    Y_balanced_ecg.npy  — shape (N,)  0=Normal, 1=MI
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

from config import SEED, DEVICE, BATCH_SIZE


class ECGDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def to_tensor(X: np.ndarray, y: np.ndarray):
    """Transpose to (N, 12, T) — CNN input format."""
    return (
        torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def load_data(x_path: str = 'X_balanced_ecg.npy',
              y_path: str = 'Y_balanced_ecg.npy'):
    """
    Load raw arrays, perform 60/20/20 stratified split, and per-lead z-score
    normalisation (fit on train only).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    Xte, yte                              : raw test tensors for inference
    criterion                             : class-weighted CrossEntropyLoss
    counts                                : Counter of class frequencies
    """
    X = np.load(x_path)   # (N, 1000, 12)
    Y = np.load(y_path)   # (N,)

    counts = Counter(Y.tolist())
    print(f'Dataset: {X.shape}  |  Normal: {counts[0]}  |  MI: {counts[1]}')
    print(f'Class imbalance ratio: 1 : {counts[0] / counts[1]:.1f}')

    # ── 60 / 20 / 20 stratified split ────────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, Y, test_size=0.20, stratify=Y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=SEED)

    # ── Per-lead z-score normalisation (fit on train only) ────────────────
    mu  = X_train.mean(axis=(0, 1), keepdims=True)   # (1, 1, 12)
    sig = X_train.std (axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mu) / sig
    X_val   = (X_val   - mu) / sig
    X_test  = (X_test  - mu) / sig

    Xtr, ytr = to_tensor(X_train, y_train)
    Xva, yva = to_tensor(X_val,   y_val)
    Xte, yte = to_tensor(X_test,  y_test)

    train_loader = DataLoader(
        ECGDataset(Xtr, ytr), batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(
        ECGDataset(Xva, yva), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0)
    test_loader  = DataLoader(
        ECGDataset(Xte, yte), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0)

    # ── Class-balanced loss weight ────────────────────────────────────────
    import torch.nn as nn
    n0, n1 = counts[0], counts[1]
    cw = torch.tensor(
        [len(Y) / (2 * n0), len(Y) / (2 * n1)], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    print(f'Class weights → Normal: {cw[0]:.4f}  MI: {cw[1]:.4f}')
    print(f'Train: {len(ytr):,}  Val: {len(yva):,}  Test: {len(yte):,}')

    return train_loader, val_loader, test_loader, Xte, yte, criterion, counts


if __name__ == '__main__':
    train_loader, val_loader, test_loader, Xte, yte, criterion, counts = load_data()
