"""
evaluate.py — Threshold optimisation, metric computation, and result reporting.

Public API
----------
find_optimal_threshold : Sweep thresholds for best F1 / Youden index
compute_metrics        : Full metric report on any DataLoader
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from config import DEVICE
from train import eval_epoch


# ─── Threshold optimisation ───────────────────────────────────────────────────

def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray,
                            metric: str = 'f1'):
    """
    Sweep thresholds in [0.1, 0.9] and return the one maximising *metric*.

    Parameters
    ----------
    labels : true binary labels
    probs  : predicted positive-class probabilities
    metric : 'f1' or 'youden'

    Returns
    -------
    best_threshold : float
    best_score     : float
    """
    thresholds = np.linspace(0.1, 0.9, 161)
    scores = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        if metric == 'f1':
            scores.append(f1_score(labels, preds, zero_division=0))
        elif metric == 'youden':
            sens = recall_score(labels, preds, zero_division=0)
            spec = (
                ((preds == 0) & (labels == 0)).sum() / (labels == 0).sum()
                if (labels == 0).sum() > 0 else 0
            )
            scores.append(sens + spec - 1)
    best_t = thresholds[np.argmax(scores)]
    return best_t, max(scores)


# ─── Metric computation ───────────────────────────────────────────────────────

def compute_metrics(model, test_loader, val_loader, criterion,
                    device=DEVICE, name='Model', lead_weights=None):
    """
    Evaluate *model* on *test_loader*.  Threshold is tuned on *val_loader*.

    Returns
    -------
    metrics_opt : dict of metric → float (at optimal threshold)
    labels      : np.ndarray of true labels
    probs       : np.ndarray of predicted MI probabilities
    opt_t       : optimal threshold
    """
    _, _, probs, labels = eval_epoch(
        model, test_loader, criterion, device, lead_weights)

    _, _, val_probs, val_labels = eval_epoch(
        model, val_loader, criterion, device, lead_weights)
    opt_t, _ = find_optimal_threshold(val_labels, val_probs, metric='f1')

    preds_05  = (probs >= 0.50).astype(int)
    preds_opt = (probs >= opt_t ).astype(int)

    def _row(preds):
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        return {
            'Accuracy'   : accuracy_score(labels, preds),
            'Precision'  : precision_score(labels, preds, zero_division=0),
            'Recall/Sens': recall_score(labels, preds, zero_division=0),
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1'         : f1_score(labels, preds, zero_division=0),
            'AUC-ROC'    : roc_auc_score(labels, probs),
        }

    m_05  = _row(preds_05)
    m_opt = _row(preds_opt)

    print(f'\n{"─"*60}')
    print(f'  {name}')
    print(f'{"─"*60}')
    print(f'  Optimal threshold (F1): {opt_t:.3f}')
    print(f'{"Metric":<16} {"t=0.50":>10} {f"t={opt_t:.2f}":>10}')
    print('─' * 40)
    for k in m_05:
        print(f'{k:<16} {m_05[k]:>10.4f} {m_opt[k]:>10.4f}')

    return m_opt, labels, probs, opt_t
