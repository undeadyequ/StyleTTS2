# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
from typing import Optional, Literal, Tuple

import torch
import torch.nn.functional as F


def clamp_delta_per_channel(x: torch.Tensor, max_step=0.05):
    """
    x: [B, T, C]
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x as [B,T,C], got shape {tuple(x.shape)}")

    y = x.clone()
    for t in range(1, x.shape[1]):
        lo = y[:, t - 1, :] - max_step
        hi = y[:, t - 1, :] + max_step
        y[:, t, :] = torch.max(torch.min(y[:, t, :], hi), lo)
    return y


def temporal_ema_per_channel(x: torch.Tensor, alpha=0.2):
    """
    x: [B, T, C]
    returns: [B, T, C]
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x as [B,T,C], got shape {tuple(x.shape)}")

    y = torch.zeros_like(x)
    y[:, 0, :] = x[:, 0, :]
    for t in range(1, x.shape[1]):
        y[:, t, :] = (1 - alpha) * y[:, t - 1, :] + alpha * x[:, t, :]
    return y


def clamp_delta(x, max_step=0.05):
    """
    x: [B, T, 1]
    """
    y = x.clone()
    for t in range(1, x.shape[1]):
        y[:, t] = torch.clamp(
            y[:, t],
            y[:, t - 1] - max_step,
            y[:, t - 1] + max_step,
        )
    return y


def temporal_ema(x, alpha=0.2):
    """
    x: [B, T, 1]
    """
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        y[:, t] = (1 - alpha) * y[:, t - 1] + alpha * x[:, t]
    return y


class EMAQuantile:
    def __init__(self, q=0.9, alpha=0.1):
        self.q = q
        self.alpha = alpha
        self.value = None

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """
        x must be detached before calling
        """
        q_val = torch.quantile(x.float(), self.q).item()
        if self.value is None:
            self.value = q_val
        else:
            self.value = self.alpha * q_val + (1 - self.alpha) * self.value
        return self.value

class EMAQuantilePerChannel:
    """
    Keeps EMA of per-channel quantile.
    For x shaped [B, T, C], returns m shaped [C].
    """
    def __init__(self, q=0.9, alpha=0.1, num_channels=2):
        self.q = q
        self.alpha = alpha
        self.num_channels = num_channels
        self.value = None  # tensor [C]

    @torch.no_grad()
    def update(self, x_detached: torch.Tensor):
        """
        x_detached: [B, T, C], MUST be detached before calling.
        returns: m_ema [C]
        """
        if x_detached.dim() != 3:
            raise ValueError(f"Expected x as [B,T,C], got shape {tuple(x_detached.shape)}")
        C = x_detached.shape[-1]
        if C != self.num_channels:
            raise ValueError(f"Expected C={self.num_channels}, got C={C}")

        # Quantile per channel over (B,T) -> q_val: [C]
        q_val = torch.quantile(x_detached.float().reshape(-1, C), self.q, dim=0)

        if self.value is None:
            self.value = q_val
        else:
            self.value = self.alpha * q_val + (1 - self.alpha) * self.value
        return self.value  # [C]



def soft_cap(x: torch.Tensor, cap: float, tau: float = 0.02) -> torch.Tensor:
    # returns approx min(x, cap), smooth everywhere
    c = torch.tensor(cap, device=x.device, dtype=x.dtype)
    return c - tau * F.softplus((c - x) / tau)

def cap_pi_ref(pi_ref, pi_max: float):
    return torch.clamp(pi_ref, max=pi_max)

# -----------------------------
# Utilities
# -----------------------------
def expand_to_time(x: torch.Tensor, T: int) -> torch.Tensor:
    """[B, D] -> [B, T, D]"""
    return x.unsqueeze(1).expand(-1, T, -1)


def match_time_length_safe(
    z: torch.Tensor,
    T_target: int,
    *,
    max_crop: int = 4,
    interp_mode: Literal["linear", "nearest"] = "linear",
) -> torch.Tensor:
    """
    Match z's time length to T_target by:
      - crop/pad if abs(diff) <= max_crop
      - else interpolate along time axis

    z: [B, T, D]
    returns: [B, T_target, D]
    """
    assert z.ndim == 3, f"Expected [B,T,D], got {tuple(z.shape)}"
    B, T, D = z.shape
    if T == T_target:
        return z

    # crop and pad if slight different
    diff = T - T_target
    if abs(diff) <= max_crop:
        if diff > 0:
            # center-crop
            start = diff // 2
            return z[:, start : start + T_target, :]
        else:
            # replicate-pad
            pad_total = -diff
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            # pad format for 3D [B,T,D] is (D_left,D_right,T_left,T_right)
            return F.pad(z, (0, 0, pad_left, pad_right), mode="replicate")

    # interpolate
    x = z.transpose(1, 2)  # [B, D, T]
    if interp_mode == "linear":
        x = F.interpolate(x, size=T_target, mode="linear", align_corners=False)
    elif interp_mode == "nearest":
        x = F.interpolate(x, size=T_target, mode="nearest")
    else:
        raise ValueError(f"Unsupported interp_mode={interp_mode}")
    return x.transpose(1, 2)  # [B, T_target, D]