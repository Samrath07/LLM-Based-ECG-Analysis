"""
visualize.py — Plotting utilities for the ECG MI detection project.

Functions
---------
plot_training_curves         : Loss + AUC + ROC comparison
plot_confusion_matrices      : Side-by-side confusion matrices
visualise_patient_prediction : Full clinical dashboard for one patient
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import recall_score

from config import LEAD_NAMES


# ─── Training curves & ROC ────────────────────────────────────────────────────

def plot_training_curves(hist_llm, hist_base, labels, probs_llm, probs_base,
                          m_llm, m_base, save: str = 'training_curves.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    ax.plot(hist_llm ['train_loss'], label='LLM Train',  color='steelblue',  lw=2)
    ax.plot(hist_llm ['val_loss'],   label='LLM Val',    color='steelblue',  lw=2, ls='--')
    ax.plot(hist_base['train_loss'], label='Base Train', color='darkorange', lw=2)
    ax.plot(hist_base['val_loss'],   label='Base Val',   color='darkorange', lw=2, ls='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # AUC
    ax = axes[1]
    ax.plot(hist_llm ['val_auc'], label='LLM  AUC', color='steelblue',  lw=2)
    ax.plot(hist_base['val_auc'], label='Base AUC', color='darkorange', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val AUC-ROC')
    ax.set_title('Validation AUC-ROC', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # ROC
    ax = axes[2]
    for probs, label, color in [
        (probs_llm,  f'1D-CNN+LLM  AUC={m_llm["AUC-ROC"]:.4f}',  'steelblue'),
        (probs_base, f'1D-CNN Base AUC={m_base["AUC-ROC"]:.4f}',  'darkorange'),
    ]:
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, label=label, lw=2, color=color)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Test Set', fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)

    plt.suptitle('Pure 1D-CNN + LLM Dynamic Attention — ECG MI Detection',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {save}')


# ─── Confusion matrices ───────────────────────────────────────────────────────

def _plot_cm(labels, probs, threshold, title, ax):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'MI'], yticklabels=['Normal', 'MI'])
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('True')
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ax.set_xlabel(f'Predicted  |  Sens: {sens:.3f}  Spec: {spec:.3f}')


def plot_confusion_matrices(labels, probs_llm, t_llm, probs_base, t_base,
                             save: str = 'confusion_matrices.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_cm(labels, probs_llm,  t_llm,  f'LLM 1D-CNN  (t={t_llm:.2f})',  ax1)
    _plot_cm(labels, probs_base, t_base, f'Baseline CNN (t={t_base:.2f})', ax2)
    plt.suptitle('Confusion Matrices — Test Set', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {save}')


# ─── Clinical dashboard ───────────────────────────────────────────────────────

def visualise_patient_prediction(ecg_signal, result, patient,
                                  true_label=None, save: str = None):
    """
    Full clinical dashboard:
      Rows 0-2 : 12-lead ECG strips colour-coded by attention weight
      Row  3   : lead-weight bar chart | risk gauge | patient summary
    """
    lw        = result['lead_weights']
    mi_prob   = result['prob_mi']
    pred      = result['prediction']
    title_col = '#B71C1C' if 'INFARCTION' in pred else '#1B5E20'

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.40, wspace=0.35)

    fig.suptitle(
        f"{patient.get('age')}y {patient.get('gender').title()}  │  "
        f"Prediction: {pred}  ({mi_prob:.1%} MI probability)  │  "
        f"Method: {result['attention_method']}",
        fontsize=13, fontweight='bold', color=title_col,
    )

    lead_groups  = [['I','II','III','aVR'], ['aVL','aVF','V1','V2'], ['V3','V4','V5','V6']]
    w_arr        = np.array([lw[l] for l in LEAD_NAMES])
    lead_colors  = plt.cm.RdYlGn(w_arr)
    lead_color_map = dict(zip(LEAD_NAMES, lead_colors))

    for row_i, group in enumerate(lead_groups):
        for col_i, lead in enumerate(group):
            ax  = fig.add_subplot(gs[row_i, col_i])
            idx = LEAD_NAMES.index(lead)
            sig = ecg_signal[idx]
            lc  = lead_color_map[lead]
            ax.plot(sig, color=lc[:3], lw=0.9)
            ax.fill_between(range(len(sig)), sig, alpha=0.12, color=lc[:3])
            w_val = lw[lead]
            ax.set_title(
                f'{lead}  (w={w_val:.3f})', fontsize=10, fontweight='bold',
                color='#C62828' if w_val > 0.08 else '#1565C0' if w_val > 0.05 else '#757575',
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1 + w_val * 20)
                spine.set_color(lc[:3])

    # Lead-weight bar chart
    ax     = fig.add_subplot(gs[3, :2])
    w_vals = [lw[l] for l in LEAD_NAMES]
    bar_cols = plt.cm.RdYlGn(np.array(w_vals) / max(w_vals))
    ax.bar(LEAD_NAMES, w_vals, color=bar_cols, edgecolor='black', lw=0.7)
    top3_idx = np.argsort(w_vals)[-3:]
    for i in top3_idx:
        ax.text(i, w_vals[i] + 0.003, '★', ha='center', fontsize=12, color='gold')
    ax.axhline(1/12, color='gray', ls='--', lw=1, label='Uniform')
    ax.set_ylabel('Attention Weight')
    ax.set_title('LLM Lead Attention', fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=9)

    # Risk gauge
    ax    = fig.add_subplot(gs[3, 2])
    theta = np.linspace(np.pi, 0, 300)
    for t1, t2, c in zip(theta[:-1], theta[1:], plt.cm.RdYlGn(np.linspace(0, 1, 299))):
        ax.plot([np.cos(t1), np.cos(t2)], [np.sin(t1), np.sin(t2)],
                color=c, lw=6, solid_capstyle='round')
    na = np.pi * (1 - mi_prob)
    ax.annotate('', xy=(0.72 * np.cos(na), 0.72 * np.sin(na)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.text(0, -0.28, f'{mi_prob:.1%}\nMI Risk', ha='center',
            fontsize=13, fontweight='bold', color=title_col)
    ax.text(-1.1, -0.18, 'Normal', fontsize=8, color='#1B5E20')
    ax.text( 0.72,-0.18, 'MI',     fontsize=8, color='#B71C1C')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.4, 1.2)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Risk Gauge', fontweight='bold')

    # Patient summary
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')
    true_str = ('MI' if true_label == 1 else 'Normal') if true_label is not None else 'N/A'
    correct  = ('✓' if true_label is not None and result['class'] == true_label
                else ('✗' if true_label is not None else ''))
    lines = [
        ('Patient',    f"{patient.get('age')}y {patient.get('gender').title()}"),
        ('Symptoms',   patient.get('symptoms', '')[:50]),
        ('Risks',      ', '.join(patient.get('risk_factors', []))[:40]),
        ('History',    patient.get('history', 'N/A')[:40]),
        ('Method',     result['attention_method']),
        ('Top Leads',  str(result['top3_leads'])),
        ('PREDICTION', pred),
        ('MI Prob',    f"{mi_prob:.2%}"),
        ('True Label', f"{true_str} {correct}"),
    ]
    for i, (k, v) in enumerate(lines):
        fw = 'bold' if k == 'PREDICTION' else 'normal'
        co = (title_col if k == 'PREDICTION'
              else 'green' if correct == '✓' and k == 'True Label'
              else 'red'   if correct == '✗' and k == 'True Label'
              else 'black')
        ax.text(0.02, 0.97 - i * 0.10, f'{k}:', transform=ax.transAxes,
                fontsize=9, fontweight='bold')
        ax.text(0.40, 0.97 - i * 0.10, v,       transform=ax.transAxes,
                fontsize=9, color=co, fontweight=fw)
    ax.set_title('Clinical Summary', fontweight='bold')

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f'Saved → {save}')
    plt.show()
