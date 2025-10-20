# References:
# https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/transformer.py
# https://github.com/jaywalnut310/vits/blob/main/attentions.py
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from drawspeech.utilities.vis import save_plot
import math
from drawspeech.modules.ditmodules.blockAttention import MultiHeadAttention, MultiHeadAttentionCross
from drawspeech.utilities.mask import sequence_mask

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

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


# modified from https://github.com/sh-lee-prml/HierSpeechpp/blob/main/modules.py#L390
class DiTConVBlockCross(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0, adaln_n=9, phoneme_RoPE="frame"):
        super().__init__()
        self.adaln = adaln_n
        # self
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn = MultiHeadAttention(hidden_channels, hidden_channels, num_heads, p_dropout)

        # cross
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn2 = MultiHeadAttentionCross(hidden_channels, hidden_channels, num_heads, p_dropout, phoneme_RoPE)
        self.norm2_cond = nn.LayerNorm(hidden_channels, elementwise_affine=False)

        self.norm3 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.mlp = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(gin_channels, hidden_channels) if gin_channels != hidden_channels else nn.Identity(),
            nn.SiLU(),
            #nn.Linear(hidden_channels, 6 * hidden_channels, bias=True)
            nn.Linear(hidden_channels, adaln_n * hidden_channels, bias=True)
        )
        """
        self.pitch_gate = nn.Sequential(
            nn.SiLU(),
            #nn.Linear(hidden_channels, 1, bias=True)
            nn.Linear(hidden_channels, 9 * hidden_channels, bias=True)
        )
        """

    def forward(self, x, c, x_mask, r, p_mask, q_f_pos=None, k_f_pos=None):
        """
        Args:
            x : [batch_size, channel, time]
            c : [batch_size, channel]
            x_mask : [batch_size, 1, time]
            r: [batch_size, channel, time]
        return the same shape as x
        """

        x = x * x_mask
        attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1)  # shape: [batch_size, 1, time, time]
        attn_mask = torch.zeros_like(attn_mask).masked_fill(attn_mask == 0, -torch.finfo(x.dtype).max)

        # ref and x may be different
        if r is not None:
            """
            # when use mels with no hop200
            r_length = torch.tensor([r.size(-1)], dtype=torch.long, device=r.device)
            r_mask = sequence_mask(r_length, r.size(2)).unsqueeze(1).to(r.dtype)
            attn_cross_mask = r_mask.unsqueeze(1) * x_mask.unsqueeze(-1)  # shape: [batch_size, 1, time, time]
            attn_cross_mask = torch.zeros_like(attn_cross_mask).masked_fill(attn_cross_mask == 0, -torch.finfo(x.dtype).max)
            """
            attn_cross_mask = p_mask.unsqueeze(1) * x_mask.unsqueeze(-1)  # shape: [batch_size, 1, time, time]
            attn_cross_mask = torch.zeros_like(attn_cross_mask).masked_fill(attn_cross_mask == 0, -torch.finfo(x.dtype).max)
        if self.adaln == 9:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(2).chunk(9, dim=1)  # shape: [batch_size, channel, 1]
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(2).chunk(6, dim=1)  #########TEMP shape: [batch_size, channel, 1]

        # selfAttn
        x = self.modulate(self.norm1(x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa)
        attn_out, _ = self.attn(x, attn_mask)
        x = x + gate_msa * attn_out * x_mask

        # CrossAttn
        if self.adaln == 9:
            x = self.modulate(self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mca, scale_mca) # LN, scale, shift
        else:
            x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        r = self.norm2_cond(r.transpose(1, 2)).transpose(1, 2)
        x_cross, attn_map = self.attn2(x=x, c=r, attn_mask=attn_cross_mask, q_f_pos=q_f_pos, k_f_pos=k_f_pos,
                                       refenh_ind_dur=None, synenh_ind_dur=None)  # (b) attn_mask = attn_cross_mask
        if self.adaln == 9:
            x = x + gate_mca * x_cross * x_mask
        else:
            x = x + x_cross * x_mask

        # mlp
        x = x + gate_mlp * self.mlp(self.modulate(self.norm3(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp), x_mask)
        #save_plot(attn_map[0, 0].detach().cpu(), "attn_map1.png")
        return x, attn_map

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift


