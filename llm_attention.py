"""
llm_attention.py — LLM-guided and rule-based lead weight helpers.

Provides
--------
SYSTEM_PROMPT           : Clinical prompt sent to Claude
build_patient_prompt    : Format a patient dict into a prompt string
query_llm_lead_weights  : Call the Anthropic API; returns raw weight dict
weights_to_tensor       : Softmax-sharpen raw weights → Tensor
get_uniform_weights     : Uniform 1/12 weight vector
rule_based_lead_weights : Fast offline rule-based fallback
"""

import json
import re

import requests
import torch

from config import LEAD_NAMES, N_LEADS, DEVICE, ANTHROPIC_API_URL, ANTHROPIC_MODEL


# ─── LLM helpers ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical ECG expert assisting a deep learning MI detection model.
Given a patient profile, assign importance weights (0.0–1.0) to each of the 12 ECG leads.

ECG Territory Guide:
- Inferior MI  (RCA occlusion) : II, III, aVF primary; reciprocal in I, aVL
- Anterior MI  (LAD occlusion) : V1–V4 primary; I, aVL secondary
- Lateral MI   (LCx occlusion) : I, aVL, V5, V6 primary
- Posterior MI (reciprocal)    : V1, V2 (tall R, ST depression)
- RV MI        (proximal RCA)  : V1, aVR elevated
- Septal MI                    : V1, V2 primarily

Consider age, gender, symptom pattern, risk factors, and prior history.
Higher weight = more diagnostically important for this patient.

RESPOND ONLY WITH VALID JSON. No markdown, no explanation, no extra text.
Exact format:
{"I": 0.0, "II": 0.0, "III": 0.0, "aVR": 0.0, "aVL": 0.0, "aVF": 0.0,
 "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0, "V6": 0.0}"""


def build_patient_prompt(patient: dict) -> str:
    age          = patient.get('age', 55)
    gender       = patient.get('gender', 'male')
    symptoms     = patient.get('symptoms', 'chest pain')
    risk_factors = patient.get('risk_factors', [])
    history      = patient.get('history', 'none')
    rf_str       = ', '.join(risk_factors) if risk_factors else 'none'
    return (
        f"Patient ECG Analysis Request:\n"
        f"Age: {age}y | Gender: {gender}\n"
        f"Symptoms: {symptoms}\n"
        f"Risk factors: {rf_str}\n"
        f"History: {history}\n\n"
        f"Assign importance weights 0.0–1.0 to all 12 leads for MI detection in this patient."
    )


def query_llm_lead_weights(patient: dict, api_key: str,
                            temperature: float = 0.1) -> dict:
    """Query Claude for patient-specific lead weights. Returns raw dict."""
    headers = {
        'x-api-key'        : api_key,
        'anthropic-version': '2023-06-01',
        'content-type'     : 'application/json',
    }
    payload = {
        'model'      : ANTHROPIC_MODEL,
        'max_tokens' : 256,
        'temperature': temperature,
        'system'     : SYSTEM_PROMPT,
        'messages'   : [{'role': 'user', 'content': build_patient_prompt(patient)}],
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers,
                         json=payload, timeout=30)
    resp.raise_for_status()
    raw_text = resp.json()['content'][0]['text'].strip()
    match = re.search(r'\{[^{}]+\}', raw_text, re.DOTALL)
    if not match:
        raise ValueError(f'Claude did not return valid JSON. Got: {raw_text}')
    raw_weights = json.loads(match.group())
    for lead in LEAD_NAMES:
        if lead not in raw_weights:
            raw_weights[lead] = 0.5
    return raw_weights


def weights_to_tensor(raw_weights: dict, temperature: float = 5.0,
                       device=DEVICE) -> torch.Tensor:
    """
    Softmax-normalise raw lead weights with a sharpening temperature.
    temperature > 1 → more peaked (highlights top leads more aggressively).
    """
    w = torch.tensor([raw_weights[l] for l in LEAD_NAMES], dtype=torch.float32)
    return torch.softmax(w * temperature, dim=0).to(device)


def get_uniform_weights(device=DEVICE) -> torch.Tensor:
    return (torch.ones(N_LEADS, dtype=torch.float32) / N_LEADS).to(device)


# ─── Rule-based fallback ──────────────────────────────────────────────────────

CLINICAL_RULES = {
    'inferior' : {'II': 0.95, 'III': 0.92, 'aVF': 0.90,
                  'I': 0.30, 'aVL': 0.25, 'V1': 0.40,
                  'V2': 0.35, 'V3': 0.35, 'V4': 0.35, 'V5': 0.40, 'V6': 0.40, 'aVR': 0.30},
    'anterior' : {'V1': 0.90, 'V2': 0.95, 'V3': 0.95, 'V4': 0.90,
                  'V5': 0.60, 'V6': 0.50, 'I': 0.50, 'aVL': 0.55,
                  'II': 0.35, 'III': 0.25, 'aVF': 0.30, 'aVR': 0.40},
    'lateral'  : {'I': 0.92, 'aVL': 0.90, 'V5': 0.88, 'V6': 0.85,
                  'V4': 0.60, 'II': 0.40, 'III': 0.30, 'aVF': 0.35,
                  'V1': 0.30, 'V2': 0.35, 'V3': 0.45, 'aVR': 0.25},
    'posterior': {'V1': 0.90, 'V2': 0.88, 'V3': 0.70, 'V4': 0.60,
                  'II': 0.55, 'III': 0.55, 'aVF': 0.55, 'V5': 0.50,
                  'V6': 0.45, 'I': 0.40, 'aVL': 0.35, 'aVR': 0.50},
    'septal'   : {'V1': 0.93, 'V2': 0.91, 'V3': 0.75, 'V4': 0.60,
                  'I': 0.35, 'II': 0.35, 'III': 0.30, 'aVR': 0.45,
                  'aVL': 0.30, 'aVF': 0.30, 'V5': 0.40, 'V6': 0.35},
    'default'  : {l: 1 / N_LEADS for l in LEAD_NAMES},
}


def rule_based_lead_weights(patient: dict) -> dict:
    """
    Derive lead weights from simple keyword rules + demographic modifiers.
    Falls back to uniform 0.5 when no territory keywords match.
    """
    symptoms = patient.get('symptoms', '').lower()
    history  = patient.get('history',  '').lower()
    text     = symptoms + ' ' + history

    if any(k in text for k in ['inferior', 'rca', 'diaphragm', 'right coronary']):
        weights = dict(CLINICAL_RULES['inferior'])
    elif any(k in text for k in ['septal', 'septal mi']):
        weights = dict(CLINICAL_RULES['septal'])
    elif any(k in text for k in ['anterior', 'lad', 'left anterior', 'stemi v']):
        weights = dict(CLINICAL_RULES['anterior'])
    elif any(k in text for k in ['lateral', 'lcx', 'circumflex', 'diagonal']):
        weights = dict(CLINICAL_RULES['lateral'])
    elif any(k in text for k in ['posterior', 'reciprocal']):
        weights = dict(CLINICAL_RULES['posterior'])
    else:
        weights = {l: 0.5 for l in LEAD_NAMES}

    # Demographic modifiers
    age    = patient.get('age', 55)
    gender = patient.get('gender', 'male').lower()
    if age > 60 and gender == 'male':
        for l in ['V1', 'V2', 'V3', 'V4']:
            weights[l] = min(1.0, weights[l] + 0.12)
    elif gender == 'female':
        for l in ['I', 'aVL', 'V5', 'V6']:
            weights[l] = min(1.0, weights[l] + 0.08)

    risk = [r.lower() for r in patient.get('risk_factors', [])]
    if 'diabetes' in risk:
        for l in ['II', 'III', 'aVF']:
            weights[l] = min(1.0, weights[l] + 0.10)
    if 'hypertension' in risk:
        for l in ['V5', 'V6', 'aVL']:
            weights[l] = min(1.0, weights[l] + 0.08)

    return weights


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    demo_patient = {
        'age': 68, 'gender': 'male',
        'symptoms': 'crushing anterior chest pain radiating to left arm, ST elevation V1-V4',
        'risk_factors': ['diabetes', 'smoking', 'hypertension'],
        'history': 'previous LAD angioplasty 2 years ago',
    }
    rb = rule_based_lead_weights(demo_patient)
    print('Rule-based lead weights (anterior STEMI profile):')
    for lead, w in sorted(rb.items(), key=lambda x: -x[1]):
        bar = '█' * int(w * 20)
        print(f'  {lead:>4}: {w:.3f}  {bar}')
