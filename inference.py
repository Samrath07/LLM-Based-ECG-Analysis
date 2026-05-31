"""
inference.py — Patient-level ECG inference with LLM or rule-based lead weights.

Public API
----------
predict_patient : Run full inference pipeline for a single patient ECG
"""

import numpy as np
import torch
import torch.nn.functional as F

from config import LEAD_NAMES, DEVICE
from llm_attention import (
    query_llm_lead_weights,
    rule_based_lead_weights,
    weights_to_tensor,
)


@torch.no_grad()
def predict_patient(
    model,
    ecg_signal: np.ndarray,    # (12, T)  or  (T, 12)
    patient:    dict,
    api_key:    str   = None,
    use_llm:    bool  = True,
    temperature: float = 5.0,
    device      = DEVICE,
) -> dict:
    """
    Predict MI probability for a single patient.

    Parameters
    ----------
    model       : trained ECG_1DCNN_LLMAttention instance
    ecg_signal  : raw ECG array, shape (12, T) or (T, 12)
    patient     : dict with keys age, gender, symptoms, risk_factors, history
    api_key     : Anthropic API key (required if use_llm=True)
    use_llm     : whether to call the Claude API for lead weights
    temperature : softmax sharpening temperature for lead weights
    device      : torch.device

    Returns
    -------
    dict with prediction, class probabilities, lead weights, top-3 leads, etc.
    """
    model.eval()

    # ── Determine lead weights ─────────────────────────────────────────────
    raw_weights = None
    method      = 'uniform'

    if use_llm and api_key:
        try:
            raw_weights = query_llm_lead_weights(patient, api_key)
            method      = 'LLM (Claude)'
        except Exception as e:
            print(f'  LLM query failed ({e}). Falling back to rule-based.')

    if raw_weights is None:
        raw_weights = rule_based_lead_weights(patient)
        method      = 'Rule-based (offline)'

    lead_weights = weights_to_tensor(raw_weights, temperature=temperature,
                                     device=device)

    # ── Prepare signal ─────────────────────────────────────────────────────
    if ecg_signal.shape[0] != 12:
        ecg_signal = ecg_signal.T    # (T, 12) → (12, T)
    x = (torch.tensor(ecg_signal, dtype=torch.float32)
         .unsqueeze(0).to(device))  # (1, 12, T)

    # ── Inject LLM weights & predict ──────────────────────────────────────
    model.set_patient_weights(lead_weights)
    logits   = model(x)
    probs    = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_cls = int(probs.argmax())
    model.clear_patient_weights()

    # ── Format output ──────────────────────────────────────────────────────
    lw   = {l: float(lead_weights[i].cpu()) for i, l in enumerate(LEAD_NAMES)}
    top3 = sorted(lw.items(), key=lambda kv: -kv[1])[:3]

    return {
        'prediction'      : 'MYOCARDIAL INFARCTION' if pred_cls == 1 else 'NORMAL SINUS RHYTHM',
        'class'           : pred_cls,
        'prob_normal'     : float(probs[0]),
        'prob_mi'         : float(probs[1]),
        'lead_weights'    : lw,
        'top3_leads'      : [l for l, _ in top3],
        'attention_method': method,
        'raw_weights'     : raw_weights,
    }
