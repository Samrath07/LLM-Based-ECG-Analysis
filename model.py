"""
model.py — 1D-CNN architecture with LLM-guided dynamic attention for ECG MI detection.

Key classes
-----------
ConvBNReLU          : Conv1d → BN → ReLU building block
SqueezeExcite       : SE gate with optional LLM projection
CBAM_Temporal       : Temporal (spatial) attention arm of CBAM
DilatedResBlock     : Dilated depthwise-separable residual block
MultiScaleStatPool  : GAP + GMP + Std + Skew global pooling
ECG_1DCNN_LLMAttention : Full model with LLM-guided SE gate
ECG_1DCNN_Baseline  : Same architecture without LLM gate (ablation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_LEADS, DEVICE


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv1d → BN → ReLU."""
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dilation=1, groups=1):
        super().__init__()
        pad = (kernel + (kernel - 1) * (dilation - 1)) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=pad, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation block with optional LLM lead-weight modulation.

    Parameters
    ----------
    channels  : number of feature-map channels
    reduction : bottleneck ratio for the SE FC layers
    n_leads   : number of ECG leads (for LLM projection input dimension)
    """
    def __init__(self, channels, reduction=16, n_leads=12):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # LLM projection: 12-lead weights → channel scale vector
        self.llm_proj = nn.Sequential(
            nn.Linear(n_leads, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, channels),
            nn.Sigmoid(),
        )
        self._llm_weights = None

    def set_llm_weights(self, w: torch.Tensor):
        self._llm_weights = w

    def clear_llm_weights(self):
        self._llm_weights = None

    def forward(self, x):
        B, C, _ = x.shape
        # Standard SE path
        s    = self.gap(x).squeeze(-1)   # (B, C)
        gate = self.fc(s)                # (B, C)

        # LLM modulation (optional at inference)
        if self._llm_weights is not None:
            w_in  = self._llm_weights.unsqueeze(0).expand(B, -1)  # (B, 12)
            llm_g = self.llm_proj(w_in)                           # (B, C)
            gate  = gate + llm_g   # additive fusion → stable gradients

        gate = torch.sigmoid(gate).unsqueeze(-1)   # (B, C, 1)
        return x * gate


class CBAM_Temporal(nn.Module):
    """Temporal (spatial) arm of CBAM for 1D signals."""
    def __init__(self, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(2, 1, kernel, padding=pad, bias=False)
        self.bn   = nn.BatchNorm1d(1)

    def forward(self, x):
        avg_t = x.mean(dim=1, keepdim=True)    # (B, 1, T)
        max_t = x.max (dim=1, keepdim=True)[0]
        pool  = torch.cat([avg_t, max_t], dim=1)  # (B, 2, T)
        gate  = torch.sigmoid(self.bn(self.conv(pool)))  # (B, 1, T)
        return x * gate


class DilatedResBlock(nn.Module):
    """
    Residual block with:
      - Two dilated depthwise-separable Conv1D layers
      - Squeeze-Excitation with optional LLM gate
      - Optional downsampling via stride-2 conv
    """
    def __init__(self, in_ch, out_ch, dilation=1, stride=2,
                 dropout=0.2, use_llm_se=False, n_leads=12):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch,  out_ch, kernel=7, dilation=dilation)
        self.conv2 = ConvBNReLU(out_ch, out_ch, kernel=5, dilation=1)
        self.drop  = nn.Dropout(dropout)
        self.se    = (SqueezeExcite(out_ch, reduction=16, n_leads=n_leads)
                      if use_llm_se else SqueezeExcite(out_ch, reduction=16))

        # Downsampling shortcut
        if stride > 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool1d(stride, ceil_mode=True) if stride > 1 else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop(out)
        out = self.se(out)
        out = self.pool(out)
        if out.size(-1) != identity.size(-1):
            min_len  = min(out.size(-1), identity.size(-1))
            out      = out     [:, :, :min_len]
            identity = identity[:, :, :min_len]
        return F.relu(out + self.shortcut(x), inplace=True)


class MultiScaleStatPool(nn.Module):
    """Concatenate GAP, GMP, Std, and Skew for richer global features."""
    def forward(self, x):
        mean = x.mean(dim=-1)
        mx   = x.max (dim=-1)[0]
        std  = x.std (dim=-1)
        diff = x - mean.unsqueeze(-1)
        skew = (diff ** 3).mean(dim=-1) / (std ** 3 + 1e-8)
        return torch.cat([mean, mx, std, skew], dim=1)   # (B, 4*C)


# ─── Main Model ───────────────────────────────────────────────────────────────

class ECG_1DCNN_LLMAttention(nn.Module):
    """
    ECG MI Detector: Pure 1D-CNN with LLM-Guided Dynamic Attention.

    No LSTM — purely convolutional with dilated residual blocks.
    LLM patient weights are injected into the SE gate of block2.
    CBAM temporal attention is applied at the final block.
    """
    def __init__(self, in_channels=12, num_classes=2, base_ch=64, n_leads=12):
        super().__init__()
        self.n_leads = n_leads

        # Stem: broad kernel to capture QRS complex shape
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),       # (B, 64, 500)
        )

        self.block1 = DilatedResBlock(base_ch, 128, dilation=1, stride=2, dropout=0.20)
        self.block2 = DilatedResBlock(128,     256, dilation=2, stride=2, dropout=0.25,
                                      use_llm_se=True, n_leads=n_leads)  # ← LLM injected here
        self.block3 = DilatedResBlock(256,     256, dilation=4, stride=2, dropout=0.30)
        self.block4 = DilatedResBlock(256,     256, dilation=8, stride=2, dropout=0.30)

        self.cbam_temporal = CBAM_Temporal(kernel=7)
        self.mspool        = MultiScaleStatPool()

        feat_dim = 256 * 4
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    # ── LLM weight management ──────────────────────────────────────────────
    def set_patient_weights(self, weights: torch.Tensor):
        """Inject LLM attention weights into SE gate of block2."""
        self.block2.se.set_llm_weights(weights.to(next(self.parameters()).device))

    def clear_patient_weights(self):
        """Reset to uniform (training / batch evaluation mode)."""
        self.block2.se.clear_llm_weights()

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, x):
        # x: (B, 12, 1000)
        x = self.stem(x)           # (B, 64,  500)
        x = self.block1(x)         # (B, 128, 250)
        x = self.block2(x)         # (B, 256, 125) ← LLM modulation here
        x = self.block3(x)         # (B, 256,  63)
        x = self.block4(x)         # (B, 256,  32)
        x = self.cbam_temporal(x)  # (B, 256,  32) ← temporal attention
        x = self.mspool(x)         # (B, 1024)
        return self.classifier(x)  # (B, 2)


# ─── Baseline (no LLM gate) ───────────────────────────────────────────────────

class ECG_1DCNN_Baseline(nn.Module):
    """Same CNN architecture without LLM gate — used as ablation baseline."""
    def __init__(self, in_channels=12, num_classes=2, base_ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 15, padding=7, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.block1 = DilatedResBlock(base_ch, 128, dilation=1, stride=2, dropout=0.20)
        self.block2 = DilatedResBlock(128,     256, dilation=2, stride=2, dropout=0.25)
        self.block3 = DilatedResBlock(256,     256, dilation=4, stride=2, dropout=0.30)
        self.block4 = DilatedResBlock(256,     256, dilation=8, stride=2, dropout=0.30)
        self.cbam   = CBAM_Temporal(kernel=7)
        self.mspool = MultiScaleStatPool()
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(512, 128),     nn.GELU(), nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.cbam(x)
        return self.classifier(self.mspool(x))


# ── Sanity-check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _dummy     = torch.zeros(4, 12, 1000)
    llm_model  = ECG_1DCNN_LLMAttention()
    base_model = ECG_1DCNN_Baseline()

    with torch.no_grad():
        out = llm_model(_dummy)
    print(f'LLM model output shape : {out.shape}')   # (4, 2)

    total_llm  = sum(p.numel() for p in llm_model.parameters())
    total_base = sum(p.numel() for p in base_model.parameters())
    se_llm_p   = sum(p.numel() for p in llm_model.block2.se.llm_proj.parameters())

    print(f'LLM model total params : {total_llm:,}')
    print(f'Baseline total params  : {total_base:,}')
    print(f'LLM SE projection      : {se_llm_p:,} ({100 * se_llm_p / total_llm:.2f}% overhead)')
