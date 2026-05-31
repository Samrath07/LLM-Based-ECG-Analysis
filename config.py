"""
config.py — Global constants and device setup for ECG MI Detection.
"""

import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── ECG Configuration ────────────────────────────────────────────────────────
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
N_LEADS    = len(LEAD_NAMES)   # 12
SEQ_LEN    = 1000              # samples per lead

# ── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE = 64
MAX_LR     = 3e-3
EPOCHS     = 60
PATIENCE   = 12

# ── LLM / API ────────────────────────────────────────────────────────────────
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_MODEL   = 'claude-sonnet-4-20250514'

if __name__ == '__main__':
    print(f'Device  : {DEVICE}')
    print(f'PyTorch : {torch.__version__}')
    print(f'Leads   : {LEAD_NAMES}')
