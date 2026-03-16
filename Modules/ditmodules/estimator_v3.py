"""Fixed DiT estimator (v3).

Changes from estimator.py:
  - DitWrapperV3: uses DiTConVBlockCrossV3 (fixed residuals, no per-block norm2_cond).
  - DecoderV3: applies a single learnable LayerNorm (elementwise_affine=True) to
    the projected reference seq_style once, before the block stack, instead of
    the affine-free per-block norm2_cond that was stripping reference magnitude.
"""

import math
import torch
import torch.nn as nn
from Modules.ditmodules.diffusion_transformer import DiTConVBlock
from Modules.ditmodules.diffusion_transformer_cross_v3 import DiTConVBlockCrossV3
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class DitWrapperV3(nn.Module):
    """Add FiLM layer to condition time embedding to DiT (v3)."""

    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1,
                 gin_channels=0, time_channels=0, cross_attn=False, glb_t_concate=False):
        super().__init__()
        self.glb_t_concate = glb_t_concate
        if self.glb_t_concate:
            self.lr1 = nn.Linear(gin_channels + time_channels, gin_channels)
        else:
            self.time_fusion = FiLMLayer(hidden_channels, time_channels)
        self.cross_attn = cross_attn
        if self.cross_attn:
            self.block = DiTConVBlockCrossV3(hidden_channels, filter_channels, num_heads,
                                             kernel_size, p_dropout, gin_channels)
        else:
            self.conv1 = nn.Conv1d(hidden_channels + hidden_channels, hidden_channels, 1)
            self.block = DiTConVBlock(hidden_channels, filter_channels, num_heads,
                                      kernel_size, p_dropout, gin_channels)

    def forward(self, x, c, t, x_mask, seq_style, p_mask, q_f_pos=None, k_f_pos=None, regularize_attn_map=None):
        if self.glb_t_concate:
            c = self.lr1(torch.concat([c, t], dim=1))
        else:
            x = self.time_fusion(x, t) * x_mask

        if self.cross_attn:
            x, attn_map = self.block(x, c, x_mask, seq_style, p_mask,
                                     q_f_pos=q_f_pos, k_f_pos=k_f_pos,
                                     regularize_attn_map=regularize_attn_map)
        else:
            x = self.conv1(torch.concat([x, seq_style], dim=1))
            x = self.block(x, c, x_mask)
            attn_map = None
        return x, attn_map


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


class DecoderV3(nn.Module):
    """DiT decoder with a single learnable LayerNorm on the reference (v3).

    The key change vs Decoder: after projecting seq_style with crossCond_proj,
    a single nn.LayerNorm(hidden_channels) with elementwise_affine=True is applied
    once, and the result is shared across all blocks.  This replaces the
    elementwise_affine=False LayerNorm that was redundantly re-applied inside each
    DiTConVBlockCross, which discarded the reference magnitude without any learnable
    parameters to compensate.
    """

    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels,
                 filter_channels, dropout=0.1, n_layers=1, n_heads=4, kernel_size=3,
                 gin_channels=0, use_lsc=True, cross_attn=False, crossCond_channels=256,
                 glb_t_concate=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.use_lsc = use_lsc
        self.cross_attn = cross_attn
        self.glb_t_concate = glb_t_concate

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(hidden_channels, hidden_channels, filter_channels)

        self.cond_proj = nn.Sequential(
            nn.Conv1d(cond_channels, filter_channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size // 2))

        self.in_proj = nn.Conv1d(hidden_channels + noise_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            DitWrapperV3(hidden_channels, filter_channels, n_heads, kernel_size, dropout,
                         gin_channels, hidden_channels, cross_attn, glb_t_concate)
            for _ in range(n_layers)])
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

        self.crossCond_proj = nn.Conv1d(crossCond_channels, hidden_channels, kernel_size,
                                        padding=kernel_size // 2)
        # FIX: single learnable LayerNorm applied once before the block stack,
        # replacing the per-block elementwise_affine=False norm2_cond.
        self.crossCond_norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)

        if use_lsc:
            assert n_layers % 2 == 0
            self.n_lsc_layers = n_layers // 2
            self.lsc_layers = nn.ModuleList([
                nn.Conv1d(hidden_channels + hidden_channels, hidden_channels,
                          kernel_size, padding=kernel_size // 2)
                for _ in range(self.n_lsc_layers)])

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, t, x, mask, mu, c, seq_style=None, p_mask=None,
                return_attn_map=False, q_f_pos=None, k_f_pos=None, regularize_attn_map=None):
        t  = self.time_mlp(self.time_embeddings(t))
        mu = self.cond_proj(mu)

        x = torch.cat((x, mu), dim=1)
        x = self.in_proj(x)

        if self.cross_attn:
            if seq_style is None:
                raise IOError("seq_style must not be None in cross_attn mode")
            # Project and normalise reference once — [B, C, T] -> [B, T, C] -> LN -> [B, C, T]
            seq_style = self.crossCond_norm(
                self.crossCond_proj(seq_style).transpose(1, 2)
            ).transpose(1, 2)

        lsc_outputs = [] if self.use_lsc else None
        attn_maps = []
        for idx, block in enumerate(self.blocks):
            if self.use_lsc:
                if idx < self.n_lsc_layers:
                    lsc_outputs.append(x)
                else:
                    x = torch.cat((x, lsc_outputs.pop()), dim=1)
                    x = self.lsc_layers[idx - self.n_lsc_layers](x)

            x, attn_map = block(x, c, t, mask, seq_style, p_mask,
                                q_f_pos=q_f_pos, k_f_pos=k_f_pos,
                                regularize_attn_map=regularize_attn_map)
            if return_attn_map and attn_map is not None:
                attn_maps.append(attn_map)

        output = self.final_proj(x * mask)

        if return_attn_map:
            stacked = torch.stack(attn_maps, dim=0) if len(attn_maps) > 0 else None
            return output * mask, stacked
        return output * mask
