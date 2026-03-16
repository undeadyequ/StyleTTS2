"""Fixed DiT cross-attention block (v3).

Changes from diffusion_transformer_cross.py:
  1. Wrong residual connections in SA and CA fixed:
       WRONG: x = modulate(norm(x)); x = x + gate * attn(x)
       RIGHT: attn_out = attn(modulate(norm(x))); x = x + gate * attn_out
  2. norm2_cond removed from this block — it is now applied once at the
     Decoder level (estimator_v3.py), so each block receives a pre-normalised r.
     This avoids redundant per-block normalisation and allows the Decoder to use
     an affine LayerNorm (elementwise_affine=True) so the reference scale is
     preserved through learnable gamma/beta.
  3. Debug prints removed.
  4. Imports from block_attn_v3 (which fixes the inference mask and IOError bugs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Modules.ditmodules.block_attn_v3 import MultiHeadAttention, MultiHeadAttentionCross
from utilities.mask import sequence_mask


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class DiTConVBlockCrossV3(nn.Module):
    """DiT block with adaLN-Zero, self-attention, cross-attention and FFN.

    Key differences from DiTConVBlockCross:
      - Correct residual connections (see module docstring).
      - No norm2_cond: the reference r must be pre-normalised by the caller
        (DecoderV3 applies a single learnable LayerNorm once before the block stack).
    """

    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3,
                 p_dropout=0.1, gin_channels=0, adaln_n=9, phoneme_RoPE="frame"):
        super().__init__()
        self.adaln = adaln_n

        # self-attention
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn  = MultiHeadAttention(hidden_channels, hidden_channels, num_heads, p_dropout)

        # cross-attention
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn2 = MultiHeadAttentionCross(hidden_channels, hidden_channels, num_heads, p_dropout, phoneme_RoPE)
        # norm2_cond intentionally removed — applied once in DecoderV3

        # FFN
        self.norm3 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.mlp   = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(gin_channels, hidden_channels) if gin_channels != hidden_channels else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_channels, adaln_n * hidden_channels, bias=True)
        )

    def forward(self, x, c, x_mask, r, p_mask, q_f_pos=None, k_f_pos=None, regularize_attn_map=None):
        """
        Args:
            x      : [B, C, T_mel]
            c      : [B, C_gin]          adaLN conditioning (time + speaker)
            x_mask : [B, 1, T_mel]
            r      : [B, C, T_ref]       pre-normalised reference (from DecoderV3)
            p_mask : [B, 1, T_ref]
        """
        x = x * x_mask

        # self-attention mask
        attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1)
        attn_mask = torch.zeros_like(attn_mask).masked_fill(attn_mask == 0, -torch.finfo(x.dtype).max)

        # cross-attention mask
        if r is not None:
            attn_cross_mask = p_mask.unsqueeze(1) * x_mask.unsqueeze(-1)
            attn_cross_mask = torch.zeros_like(attn_cross_mask).masked_fill(
                attn_cross_mask == 0, -torch.finfo(x.dtype).max)
            if regularize_attn_map is not None:
                attn_cross_mask = attn_cross_mask + regularize_attn_map

        # adaLN parameters
        if self.adaln == 9:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).unsqueeze(2).chunk(9, dim=1))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).unsqueeze(2).chunk(6, dim=1))

        # --- Self-attention (fixed residual) ---
        attn_out, _ = self.attn(self.modulate(self.norm1(
            x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa), attn_mask)    # LN+Scale+shift
        x = x + gate_msa * attn_out * x_mask

        # --- Cross-attention (fixed residual) ---
        if self.adaln == 9:
            x_q = self.modulate(self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mca, scale_mca)   # LN+Scale+shift
        else:
            x_q = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x_cross, attn_map = self.attn2(x=x_q, c=r, attn_mask=attn_cross_mask, q_f_pos=q_f_pos, k_f_pos=k_f_pos,
            refenh_ind_dur=None, synenh_ind_dur=None)

        if self.adaln == 9:
            x = x + gate_mca * x_cross * x_mask
        else:
            x = x + x_cross * x_mask

        if self.training and self.adaln == 9:
            self._gate_mca_norm = torch.norm(gate_mca, p=2, dim=1).mean().item()

        # --- FFN (was already correct) ---
        x = x + gate_mlp * self.mlp(self.modulate(self.norm3(
            x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp), # LN+Scale+shift
            x_mask)

        return x, attn_map

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift
