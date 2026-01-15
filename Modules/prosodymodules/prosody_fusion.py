# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.prosodymodules.utils import (temporal_ema, temporal_ema_per_channel, EMAQuantile, EMAQuantilePerChannel,
                                          soft_cap, expand_to_time, cap_pi_ref, match_time_length_safe, clamp_delta)
import random
"""
MoE (2-expert) prosody fusion with FACTORIZED gate:
    pi_ref(t) = e(mu_t, s_g) * a(z_pred(t), z_ref(t))   in (0, 1)
    pi_pred(t) = 1 - pi_ref(t)
Experts:
    E_pred(t) = z_pred(t)
    E_ref(t)  = z_ref(t)
Fusion:
    z_fuse(t) = pi_pred(t) * z_pred(t) + pi_ref(t) * z_ref(t)
This is algebraically equivalent to the "eligibility-gated residual correction" form,
but implemented explicitly as a 2-expert MoE.
Includes:
- Safe length matching for z_ref -> T_pred (crop/pad for small diff; interpolate otherwise)
- Optional stop-gradient on gap features used inside a(z_pred, z_ref) to avoid chasing gap
- Inference controls via manual scaling of e and a (for style transfer strength)
"""

class FactorizedGateMoEProsodyFusion(nn.Module):
    """
    Two-expert MoE with factorized gate:

        e(t) = sigmoid(g_e([mu_t, s_g]))
        a(t) = sigmoid(g_a([z_pred, z_ref, Δ(t)]))
        pi_ref(t)  = e(t) * a(t)
        pi_pred(t) = 1 - pi_ref(t)

        z_fuse(t) = pi_pred(t) * z_pred(t) + pi_ref(t) * z_ref_aligned(t)

    Inputs:
      mu_t:   [B, T_pred, D_mu]
      s_g:    [B, D_s]
      z_pred: [B, T_pred, D_z]
      z_ref:  [B, T_ref,  D_z]  (from GT mel; may have slightly different length)

    Output:
      z_fuse: [B, T_pred, D_z]

    Optionally returns gates.
    """

    def __init__(self, d_mu: int, d_s: int, d_z: int, cfg: Optional[MoEFusionConfig] = None):
        super().__init__()
        self.cfg = cfg or MoEFusionConfig()

        # eligibility gate e(mu_t, s_g) -> [B,T,1]
        self.g_e = MLP(in_dim=d_mu + d_s, hidden_dim=self.cfg.e_hidden, out_dim=1, dropout=self.cfg.dropout)

        # correction strength gate a(z_pred, z_ref, gap) -> [B,T,1]
        gap_dim = 1 if self.cfg.gap_mode in ("l2", "cos") else 2
        self.g_a = MLP(in_dim= gap_dim, hidden_dim=self.cfg.a_hidden, out_dim=1, dropout=self.cfg.dropout)  # in_dim= d_z + gap_dim

        # bias init to keep pi_ref small at the beginning
        nn.init.constant_(self.g_e.net[-1].bias, float(self.cfg.init_bias_e))
        nn.init.constant_(self.g_a.net[-1].bias, float(self.cfg.init_bias_a))

        self.ema_q90 = EMAQuantile(q=0.9, alpha=0.1)

    @torch.no_grad()
    def _align_z_ref(self, z_ref: torch.Tensor, T_target: int) -> torch.Tensor:
        # If your z_ref is discrete indices, set interp_mode="nearest" and use embedding space instead.
        return match_time_length_safe(z_ref, T_target, max_crop=self.cfg.max_crop, interp_mode=self.cfg.interp_mode)

    def forward(
        self,
        mu_t: torch.Tensor,        # [B,T_pred,D_mu]
        s_g: torch.Tensor,         # [B,D_s]
        z_pred: torch.Tensor,      # [B,T_pred,D_z]
        z_ref: Optional[torch.Tensor],  # [B,T_ref,D_z] or None
        *,
        z_ref_available: bool = True,
        return_gates: bool = False,
    ):
        assert mu_t.ndim == 3 and z_pred.ndim == 3
        B, T_pred, _ = mu_t.shape

        # Inference without reference: passthrough
        if (not z_ref_available) or (z_ref is None):
            if return_gates:
                pi_ref = torch.zeros((B, T_pred, 1), device=mu_t.device, dtype=mu_t.dtype)
                e = torch.zeros_like(pi_ref)
                a = torch.zeros_like(pi_ref)
                return z_pred, pi_ref, e, a
            return z_pred

        assert z_ref.ndim == 3 and z_ref.size(0) == B and z_ref.size(-1) == z_pred.size(-1)

        # 1) Align z_ref length to T_pred (minimal, stable)
        z_ref = self._align_z_ref(z_ref, T_pred)  # [B,T_pred,D_z]
        # 2) Eligibility gate e(t)
        sg_time = expand_to_time(s_g, T_pred)
        e_in = torch.cat([mu_t, sg_time], dim=-1)
        e = torch.sigmoid(self.g_e(e_in))  # [B,T,1]
        e = e * float(self.cfg.manual_e_scale)
        e = torch.clamp(e, min=self.cfg.e_floor, max=self.cfg.e_ceiling)

        # 3) Correction gate a(t)
        diff = (z_ref - z_pred)
        if self.cfg.stopgrad_gap:
            diff = diff.detach()

        # GAP
        gap = gap_features(z_pred, z_ref, mode=self.cfg.gap_mode, stopgrad=self.cfg.stopgrad_gap)  # (b, 1/2, L)
        gap = temporal_ema(gap, alpha=0.2)

        # a_in
        #a_in = torch.cat([diff, gap], dim=-1)
        m = self.ema_q90.update(gap.detach())
        s = 0.3 * m + 1e-6
        a_in = (gap - m) / s

        #tau = 2  # > 1 early
        #a = torch.sigmoid(self.g_a(a_in) / s) # [B,T,1]
        a = torch.sigmoid(a_in)  # [B,T,1]

        a = a * float(self.cfg.manual_a_scale)
        a = torch.clamp(a, min=self.cfg.a_floor, max=self.cfg.a_ceiling)

        # 4) MoE weight
        #pi_ref = e * a                                               # [B,T,1]
        pi_ref = a
        pi_ref = clamp_delta(pi_ref, max_step=0.02)  # or 0.05
        pi_ref = soft_cap(pi_ref, cap=0.7, tau=0.02)       # <-- ADD cap
        pi_pred = 1.0 - pi_ref                                       # [B,T,1]

        # 5) MoE fusion
        z_fuse = pi_pred * z_pred + pi_ref * z_ref  # broadcast over D_z

        # 6) return mIndex
        et_mean, et_q25, et_q50, et_q75 = tensor_stats(e)
        at_mean, at_q25, at_q50, at_q75 = tensor_stats(a)

        pi_ref_mean, pi_ref_min, pi_ref_max, at_mean =  pi_ref.detach().mean().item(), pi_ref.detach().min().item(), pi_ref.detach().max().item(), a.detach().mean().item()
        at_min, at_max = a.detach().min().item(), a.detach().max().item()
        et_mean_tensor = e.mean()
        gap_mean, gap_q25, gap_q50, gap_q75 = tensor_stats(gap[0, :, 0]) # 0, 0 means: batch, l2_index
        gap_indexQ75 = torch.where(gap[0, :, 0] > gap_q75)[0]

        # random sliced 10
        if False:
            if gap_indexQ75.size(-1) > 10:
                start = random.randint(0, gap_indexQ75.size(-1) - 10)
                gap_indexQ75 = gap_indexQ75[start:start + 10]

        if return_gates:
            #return z_fuse, (pi_ref_mean, et_mean_tensor, at_mean, gap_mean, et_q25, et_q75, at_q25, at_q75)
            return z_fuse, (pi_ref_mean, pi_ref_min, pi_ref_max, at_min, at_mean, at_q25, at_q75, at_max, gap_mean, gap_q25, gap_q50, gap_q75, gap_indexQ75, m)

        return z_fuse


def gap_features(
    z_pred: torch.Tensor,
    z_ref: torch.Tensor,
    *,
    mode: Literal["l2", "cos", "both"] = "both",
    eps: float = 1e-8,
    stopgrad: bool = True,
) -> torch.Tensor:
    """
    Per-frame mismatch features Δ(t).
    z_pred: [B, T, 256]
    z_ref: [B, T, 256]
    Returns [B, T, D_gap], where D_gap in {1, 2}.
    """
    assert z_pred.shape == z_ref.shape
    diff = z_ref - z_pred
    if stopgrad:
        diff = diff.detach()

    feats = []
    if mode in ("l2", "both"):
        l2 = torch.sqrt(torch.clamp((diff * diff).sum(dim=-1, keepdim=True), min=eps))
        feats.append(l2)
    if mode in ("cos", "both"):
        zp = z_pred.detach() if stopgrad else z_pred
        zr = z_ref.detach() if stopgrad else z_ref
        cos = F.cosine_similarity(zp, zr, dim=-1, eps=eps).unsqueeze(-1)
        feats.append(1.0 - cos)
    return torch.cat(feats, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Config
# -----------------------------
@dataclass
class MoEFusionConfig:
    # Gate MLP sizes
    e_hidden: int = 256
    a_hidden: int = 256
    dropout: float = 0.0

    # a(z_pred, z_ref) uses these mismatch features
    gap_mode: Literal["l2", "cos", "both"] = "l2"
    stopgrad_gap: bool = True

    # Length matching
    max_crop: int = 4
    interp_mode: Literal["linear", "nearest"] = "linear"

    # Optional clamps (stability / ablations)
    e_floor: float = 0.0
    e_ceiling: float = 1.0
    a_floor: float = 0.0
    a_ceiling: float = 1.0

    # Optional manual strength control (inference knob)
    manual_e_scale: float = 1.0
    manual_a_scale: float = 1.0

    # Optional bias init: encourage pi_ref small at start (avoid ref-dominant collapse)
    init_bias_e: float = -2.0  # sigmoid ~ 0.12
    init_bias_a: float = -2.0  # sigmoid ~ 0.12

    pi_ref_max: float = 0.3


moeConfig = MoEFusionConfig(
    e_hidden=256,
    a_hidden=256,
    gap_mode="l2",
    stopgrad_gap=True,
    max_crop=4,
    interp_mode="linear",
    manual_e_scale=1.0,
    manual_a_scale=1.0,
    init_bias_e=-2.0,
    init_bias_a=-2.0,
    pi_ref_max=0.3
)

def tensor_stats(x: torch.Tensor):
    """
    x: tensor of shape [B, T, 1] or any shape
    """
    x_flat = x.detach().reshape(-1)

    q25 = torch.quantile(x_flat, 0.25).item()
    q50 = torch.quantile(x_flat, 0.50).item()
    q75 = torch.quantile(x_flat, 0.75).item()
    mean = x_flat.mean().item()
    return mean, q25, q50, q75


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    T_pred = 200
    T_ref = 206  # slightly different due to hop rounding / trimming
    D_mu, D_s, D_z = 256, 256, 128

    mu_t = torch.randn(B, T_pred, D_mu)
    s_g = torch.randn(B, D_s)
    z_pred = torch.randn(B, T_pred, D_z)
    z_ref = torch.randn(B, T_ref, D_z)

    moeConfig = MoEFusionConfig(
        e_hidden=256,
        a_hidden=256,
        gap_mode="both",
        stopgrad_gap=True,
        max_crop=4,
        interp_mode="linear",
        manual_e_scale=1.0,
        manual_a_scale=1.0,
        init_bias_e=-2.0,
        init_bias_a=-2.0,
    )

    moe = FactorizedGateMoEProsodyFusion(d_mu=D_mu, d_s=D_s, d_z=D_z, cfg=moeConfig)
    z_fuse, pi_ref, e, a, gap = moe(mu_t, s_g, z_pred, z_ref, return_gates=True, return_gap=True)
    l2_index, cos_index = 0, 1

    print("z_fuse:", z_fuse.shape)
    print("pi_ref mean:", pi_ref.mean().item(), "e mean:", e.mean().item(), "a mean:", a.mean().item())

    gap_mean = tensor_stats(gap[0, :, l2_index])[0]
    print("gap_mean:", gap_mean)