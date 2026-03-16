"""DiT estimator with masked-mel conditioning and no global speaker embedding (v4).

Changes from estimator_v3.py:
  - Global conditioning c removed entirely.
  - Timestep t (hidden_channels) is passed directly as adaLN conditioning,
    replacing both c and the FiLM time_fusion layer.
  - in_proj accepts concat(noised_x, cond, mu_proj):
      2 * noise_channels + hidden_channels  (cond has same dim as noised_x)
  - DecoderV4.forward: adds `cond` arg, drops `c`.
  - DitWrapperV4: no FiLM / glb_t_concate; passes t straight to the block.

Usage (in cfm config YAML):
    cfm_config:
        use_v4: true
        cross_attn: true
        ...
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.ditmodules.diffusion_transformer import DiTConVBlock
from Modules.ditmodules.diffusion_transformer_cross_v3 import DiTConVBlockCrossV3


# ---------------------------------------------------------------------------
# Helpers (identical to v3)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        self.film = nn.Conv1d(cond_channels, in_channels * 2, 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c.unsqueeze(2)), chunks=2, dim=1)
        return gamma * x + beta


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.SiLU(inplace=True),
            nn.Linear(filter_channels, out_channels))

    def forward(self, x):
        return self.layer(x)


# ---------------------------------------------------------------------------
# DitWrapperV4
# ---------------------------------------------------------------------------

class DitWrapperV4(nn.Module):
    """DiT wrapper where t (timestep, hidden_channels) directly drives adaLN.

    No FiLM layer, no glb_t_concate — t is the sole adaLN conditioning signal.
    gin_channels is fixed to hidden_channels so the first Linear in
    adaLN_modulation becomes nn.Identity().
    """

    def __init__(self, hidden_channels, filter_channels, num_heads,
                 kernel_size=3, p_dropout=0.1, cross_attn=False):
        super().__init__()
        self.cross_attn = cross_attn
        if self.cross_attn:
            self.block = DiTConVBlockCrossV3(
                hidden_channels, filter_channels, num_heads,
                kernel_size, p_dropout,
                gin_channels=hidden_channels)   # t has hidden_channels dims
        else:
            # For non-cross-attn: fuse seq_style via channel concat then self-attn
            self.conv1 = nn.Conv1d(hidden_channels * 2, hidden_channels, 1)
            self.block = DiTConVBlock(
                hidden_channels, filter_channels, num_heads,
                kernel_size, p_dropout,
                gin_channels=hidden_channels)

    def forward(self, x, t, x_mask, seq_style, p_mask,
                q_f_pos=None, k_f_pos=None, regularize_attn_map=None):
        """
        Args:
            x        : [B, C, T]
            t        : [B, hidden_channels]  — timestep embedding, used as adaLN c
            x_mask   : [B, 1, T]
            seq_style: [B, C, T_ref]
            p_mask   : [B, 1, T_ref]
        """
        if self.cross_attn:
            x, attn_map = self.block(
                x, t, x_mask, seq_style, p_mask,
                q_f_pos=q_f_pos, k_f_pos=k_f_pos,
                regularize_attn_map=regularize_attn_map)
        else:
            x = self.conv1(torch.cat([x, seq_style], dim=1))
            x = self.block(x, t, x_mask)   # DiTConVBlock returns tensor only
            attn_map = None
        return x, attn_map


# ---------------------------------------------------------------------------
# DecoderV4
# ---------------------------------------------------------------------------

class DecoderV4(nn.Module):
    """DiT decoder with masked-mel conditioning (v4).

    Input to in_proj = concat(noised_x, cond, mu_proj)
      - noised_x : [B, noise_channels, T]
      - cond      : [B, noise_channels, T]  masked mel (target region zeroed)
      - mu_proj   : [B, hidden_channels, T] after cond_proj(mu)

    No global c; t (timestep) drives all adaLN modulations.
    """

    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels,
                 filter_channels, dropout=0.1, n_layers=1, n_heads=4, kernel_size=3,
                 use_lsc=True, cross_attn=False, crossCond_channels=256):
        super().__init__()
        self.hidden_channels  = hidden_channels
        self.out_channels     = out_channels
        self.filter_channels  = filter_channels
        self.use_lsc          = use_lsc
        self.cross_attn       = cross_attn

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp        = TimestepEmbedding(hidden_channels, hidden_channels, filter_channels)

        # mu projection: cond_channels -> hidden_channels
        self.cond_proj = nn.Sequential(
            nn.Conv1d(cond_channels,    filter_channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels,  filter_channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels,  hidden_channels, kernel_size, padding=kernel_size // 2))

        # in_proj: [noised_x | cond | mu_proj] -> hidden_channels
        # cond shares noise_channels with noised_x
        self.in_proj = nn.Conv1d(
            2 * noise_channels + hidden_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList([
            DitWrapperV4(hidden_channels, filter_channels, n_heads,
                         kernel_size, dropout, cross_attn)
            for _ in range(n_layers)])

        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

        # Cross-attention reference processing (same as v3)
        self.crossCond_proj = nn.Conv1d(
            crossCond_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.crossCond_norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)

        if use_lsc:
            assert n_layers % 2 == 0
            self.n_lsc_layers = n_layers // 2
            self.lsc_layers = nn.ModuleList([
                nn.Conv1d(hidden_channels * 2, hidden_channels,
                          kernel_size, padding=kernel_size // 2)
                for _ in range(self.n_lsc_layers)])

        self.initialize_weights()

    def initialize_weights(self):
        """adaLN-Zero: zero-init the last linear so gates start at 0."""
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, t, x, cond, mask, mu,
                seq_style=None, p_mask=None,
                return_attn_map=False,
                q_f_pos=None, k_f_pos=None, regularize_attn_map=None):
        """
        Args:
            t        : [B]              timestep scalar per sample
            x        : [B, C, T]        noised mel
            cond     : [B, C, T]        masked mel (target region zeroed)
            mask     : [B, 1, T]
            mu       : [B, cond_ch, T]  frame-level text embedding
            seq_style: [B, cross_ch, T_ref]  pitch/energy contours
            p_mask   : [B, 1, T_ref]
        """
        t  = self.time_mlp(self.time_embeddings(t))     # [B, hidden_channels]
        mu = self.cond_proj(mu)                          # [B, hidden_channels, T]

        x = torch.cat((x, cond, mu), dim=1)             # [B, 2*C+H, T]
        x = self.in_proj(x)                             # [B, hidden_channels, T]

        if self.cross_attn:
            if seq_style is None:
                raise IOError("seq_style must not be None in cross_attn mode")
            seq_style = self.crossCond_norm(
                self.crossCond_proj(seq_style).transpose(1, 2)
            ).transpose(1, 2)

        lsc_outputs = [] if self.use_lsc else None
        attn_maps   = []

        for idx, block in enumerate(self.blocks):
            if self.use_lsc:
                if idx < self.n_lsc_layers:
                    lsc_outputs.append(x)
                else:
                    x = torch.cat((x, lsc_outputs.pop()), dim=1)
                    x = self.lsc_layers[idx - self.n_lsc_layers](x)
            x, attn_map = block(x, t, mask, seq_style, p_mask,
                                q_f_pos=q_f_pos, k_f_pos=k_f_pos,
                                regularize_attn_map=regularize_attn_map)
            if return_attn_map and attn_map is not None:
                attn_maps.append(attn_map)

        output = self.final_proj(x * mask)

        if return_attn_map:
            stacked = torch.stack(attn_maps, dim=0) if attn_maps else None
            return output * mask, stacked
        return output * mask
