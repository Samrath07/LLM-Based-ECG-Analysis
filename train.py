"""
train.py — Training and evaluation loops for the ECG MI detector.

Public API
----------
train_epoch   : One epoch of forward + backward pass
eval_epoch    : Evaluation pass returning loss, accuracy, probs, labels
train_model   : Full training loop with OneCycleLR + early stopping
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score

from config import DEVICE, MAX_LR, EPOCHS, PATIENCE


# ─── Per-epoch helpers ────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, crit, device, scaler=None):
    model.train()
    if hasattr(model, 'clear_patient_weights'):
        model.clear_patient_weights()   # uniform gate during training

    tl, correct, total = 0., 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss   = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(xb)
            loss   = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        tl      += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total   += yb.size(0)

    return tl / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, crit, device, lead_weights=None):
    model.eval()
    if hasattr(model, 'set_patient_weights') and lead_weights is not None:
        model.set_patient_weights(lead_weights)

    tl, correct, total = 0., 0, 0
    probs_all, labels_all = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits  = model(xb)
        tl      += crit(logits, yb).item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total   += yb.size(0)
        probs_all.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
        labels_all.extend(yb.cpu().numpy())

    if hasattr(model, 'clear_patient_weights'):
        model.clear_patient_weights()

    return tl / total, correct / total, np.array(probs_all), np.array(labels_all)


# ─── Full training loop ───────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, criterion, name,
                epochs=EPOCHS, max_lr=MAX_LR, patience=PATIENCE, device=DEVICE):
    """
    Train with AdamW + OneCycleLR + early stopping on validation AUC-ROC.

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader
    val_loader   : DataLoader
    criterion    : loss function
    name         : display name for logging
    epochs       : maximum training epochs
    max_lr       : peak learning rate for OneCycleLR
    patience     : epochs without AUC improvement before early stop
    device       : torch.device

    Returns
    -------
    model : loaded with best weights
    hist  : dict of training history lists
    """
    model = model.to(device)
    opt   = AdamW(model.parameters(), lr=max_lr / 25, weight_decay=1e-4)
    sched = OneCycleLR(opt, max_lr=max_lr,
                       steps_per_epoch=len(train_loader), epochs=epochs,
                       pct_start=0.1, anneal_strategy='cos')
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    hist = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : [],
        'val_auc'   : [],
    }
    best_auc, best_state, no_imp = 0.0, None, 0

    print(f"\n{'='*65}\n  Training: {name}\n{'='*65}")
    print(f"{'Ep':>4} | {'TrLoss':>8} | {'TrAcc':>7} | "
          f"{'VaLoss':>8} | {'VaAcc':>7} | {'VaAUC':>7}")
    print('-' * 55)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_l, tr_a = train_epoch(model, train_loader, opt, criterion, device, scaler)
        sched.step()
        va_l, va_a, probs, labels = eval_epoch(model, val_loader, criterion, device)
        va_auc = roc_auc_score(labels, probs)

        hist['train_loss'].append(tr_l);  hist['val_loss'].append(va_l)
        hist['train_acc' ].append(tr_a);  hist['val_acc' ].append(va_a)
        hist['val_auc'   ].append(va_auc)

        if va_auc > best_auc:
            best_auc   = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0;  tag = ' ✓'
        else:
            no_imp += 1;  tag = ''

        if ep % 5 == 0 or ep == 1:
            print(f"{ep:>4} | {tr_l:>8.4f} | {tr_a:>7.4f} | "
                  f"{va_l:>8.4f} | {va_a:>7.4f} | {va_auc:>7.4f}{tag}")

        if no_imp >= patience:
            print(f'\nEarly stop at epoch {ep}  (best AUC: {best_auc:.4f})')
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    print(f'Training time: {elapsed:.1f}s  |  Best val-AUC: {best_auc:.4f}')
    return model, hist
