"""Fixed cross-attention modules (v3).

Changes from blockAttention.py:
  1. attn_mask_operation always "add" — the "multiply" inference path in
     MultiHeadAttentionCross zeroed all valid attention weights; removed.
  2. IOError is now properly raised in the phoneme-RoPE guard.
  3. scaled_dot_product_attention simplified: attn_mask_operation param removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Modules.ditmodules.positionEmbedder import PhoneRotaryPositionalEmbeddings, RotaryPositionalEmbeddings


class MultiHeadAttentionCross(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0., phoneme_RoPE="frame"):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)

        self.phoneme_RoPE = phoneme_RoPE

        if self.phoneme_RoPE == "phone" or self.phoneme_RoPE == "sel":
            self.query_rotary_pe = PhoneRotaryPositionalEmbeddings(self.k_channels * 0.5)
            self.key_rotary_pe = PhoneRotaryPositionalEmbeddings(self.k_channels * 0.5)
        else:
            self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
            self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None, q_f_pos=None, k_f_pos=None, refenh_ind_dur=None, synenh_ind_dur=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, attn_map = self.attention(q, k, v, mask=attn_mask, q_f_pos=q_f_pos, k_f_pos=k_f_pos,
                                     refenh_ind_dur=refenh_ind_dur, synenh_ind_dur=synenh_ind_dur)
        x = self.conv_o(x)
        return x, attn_map

    def attention(self, query, key, value, mask=None, q_f_pos=None, k_f_pos=None, refenh_ind_dur=None, synenh_ind_dur=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key   = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        if self.phoneme_RoPE == "phone" or self.phoneme_RoPE == "sel":
            if q_f_pos is None or k_f_pos is None:
                raise IOError("q_f_pos / k_f_pos must not be None when phoneme_RoPE is active")
            query = self.query_rotary_pe(query, seq_dur=q_f_pos)
            key   = self.key_rotary_pe(key, seq_dur=k_f_pos)
        else:
            query = self.query_rotary_pe(query)
            key   = self.key_rotary_pe(key)

        # FIX: always use "add" masking — the original "multiply" at inference
        # zeroed all valid attention weights (attn_bias is 0 for valid positions).
        dropout_p = self.p_dropout if self.training else 0
        output, attn_map = scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=dropout_p)

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, attn_map


class MultiHeadAttention(nn.Module):
    """Self-attention — unchanged from blockAttention.py."""
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)

        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe   = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        x, attn_map = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x, attn_map

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key   = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        query = self.query_rotary_pe(query)
        key   = self.key_rotary_pe(key)

        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=self.p_dropout if self.training else 0)

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, None


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None, enable_gqa=False):
    """Standard scaled dot-product attention — always adds the mask bias."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key   = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight  = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight  = torch.softmax(attn_weight, dim=-1)
    attn_weight  = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight
