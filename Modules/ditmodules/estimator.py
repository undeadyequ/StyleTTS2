import math
import torch
import torch.nn as nn
from Modules.ditmodules.diffusion_transformer import DiTConVBlock
from Modules.ditmodules.diffusion_transformer_cross import DiTConVBlockCross
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class DitWrapper(nn.Module):
    """ add FiLM layer to condition time embedding to DiT """
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0,
                 time_channels=0, cross_attn=False, official_dit=False):
        super().__init__()
        self.official_dit = official_dit
        if self.official_dit:
            self.lr1 = nn.Linear(gin_channels + time_channels, gin_channels)
            self.conv1 = nn.Conv1d(hidden_channels + hidden_channels, hidden_channels, 1)
        else:
            self.time_fusion = FiLMLayer(hidden_channels, time_channels)
        self.cross_attn = cross_attn
        if self.cross_attn:
            self.block = DiTConVBlockCross(hidden_channels, filter_channels, num_heads, kernel_size, p_dropout, gin_channels)
        else:
            self.block = DiTConVBlock(hidden_channels, filter_channels, num_heads, kernel_size, p_dropout, gin_channels)
            
    def forward(self, x, c, t, x_mask, seq_style, p_mask, q_f_pos=None, k_f_pos=None, regularize_attn_map=None):
        if self.official_dit:
            c = self.lr1(torch.concat([c, t], dim=1))
        else:
            x = self.time_fusion(x, t) * x_mask

        if self.cross_attn:
            x, attn_map = self.block(x, c, x_mask, seq_style, p_mask, q_f_pos=q_f_pos, k_f_pos=k_f_pos, regularize_attn_map=regularize_attn_map)
        else:
            x = self.conv1(torch.concat([x, seq_style], dim=1))
            x = self.block(x, c, x_mask)
            attn_map = None
        return x, attn_map

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """
    def __init__(self, in_channels, cond_channels):
        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(cond_channels, in_channels * 2, 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c.unsqueeze(2)), chunks=2, dim=1)
        return gamma * x + beta
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

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

# reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class Decoder(nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, dropout=0.1, n_layers=1,
                 n_heads=4, kernel_size=3, gin_channels=0, use_lsc=True, cross_attn=False, crossCond_channels=256, official_dit=False):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels

        if official_dit:
            self.use_lsc = False
            self.cross_attn = False
        else:
            self.use_lsc = use_lsc # whether to use unet-like long skip connection
            self.cross_attn = cross_attn

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(hidden_channels, hidden_channels, filter_channels)

        # prenet for mu(encoder output)
        self.cond_proj = nn.Sequential(
            nn.Conv1d(cond_channels, filter_channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2),  # add about 3M params
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size // 2))

        self.in_proj = nn.Conv1d(hidden_channels + noise_channels, hidden_channels, 1) # cat prior and xt as input
        self.blocks = nn.ModuleList([DitWrapper(hidden_channels, filter_channels, n_heads, kernel_size, dropout, gin_channels,
                                                hidden_channels, cross_attn, official_dit) for _ in range(n_layers)])  # wrapper t
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)                  #

        # prenet for reference embedder
        """
        self.crossCond_proj = nn.Sequential(
            nn.Conv1d(crossCond_channels, filter_channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2),  # add about 3M params
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        ) # OPTION: sequential
        """
        self.crossCond_proj = nn.Conv1d(crossCond_channels, hidden_channels, kernel_size, padding=kernel_size // 2)

        if use_lsc:
            assert n_layers % 2 == 0
            self.n_lsc_layers = n_layers // 2
            self.lsc_layers = nn.ModuleList([nn.Conv1d(hidden_channels + hidden_channels, hidden_channels, kernel_size, padding = kernel_size // 2) for _ in range(self.n_lsc_layers)])
            
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, t, x, mask, mu, c, seq_style=None, p_mask=None, return_attn_map=True, q_f_pos=None, k_f_pos=None,
                regularize_attn_map=None):
        """Forward pass of the DiT model.

        Args:
            t (torch.Tensor): timestep, shape (batch_size)
            x (torch.Tensor): noise, shape (batch_size, in_channels, time)
            mask (torch.Tensor): shape (batch_size, 1, time)
            mu (torch.Tensor): output of encoder, shape (batch_size, in_channels, time)
            c (torch.Tensor): shape (batch_size, gin_channels)
            seq_style: (batch_size, seq_channels, time)

        Returns:
            _type_: _description_
        """
        t = self.time_mlp(self.time_embeddings(t))
        mu = self.cond_proj(mu)
        #print("mu mean_std", torch.mean(mu, dim=1), torch.std(mu, dim=1))
        #print("mu min_max", torch.min(mu, dim=1).values, torch.max(mu, dim=1).values)

        x = torch.cat((x, mu), dim=1)
        x = self.in_proj(x)

        if self.cross_attn:
            if seq_style is None:
                raise IOError("seq_syle should not be none in cross_attn mode")
            seq_style = self.crossCond_proj(seq_style)
            #print("seq mean_std", torch.mean(prosody_encoder, dim=1), torch.std(prosody_encoder, dim=1))
            #print("seq min_max", torch.min(prosody_encoder, dim=1).values, torch.max(prosody_encoder, dim=1).values)

        lsc_outputs = [] if self.use_lsc else None
        attn_maps = []
        for idx, block in enumerate(self.blocks):
            # add long skip connection, see https://arxiv.org/pdf/2209.12152 for more details
            if self.use_lsc:
                if idx < self.n_lsc_layers:
                    lsc_outputs.append(x)
                else:
                    #print(f"block: {idx}: before lsc:", torch.mean(torch.abs(x), dim=1))
                    x = torch.cat((x, lsc_outputs.pop()), dim=1)
                    x = self.lsc_layers[idx - self.n_lsc_layers](x)
            ############### Check code #######33
            #print(f"block: {idx}: before block:", torch.mean(torch.abs(x), dim=1))
            x, attn_map = block(x, c, t, mask, seq_style, p_mask, q_f_pos=q_f_pos, k_f_pos=k_f_pos, regularize_attn_map=regularize_attn_map)
            attn_maps.append(attn_map)
        output = self.final_proj(x * mask)
        if attn_maps[0] is not None:
            attn_maps = torch.stack(attn_maps, dim=0)
        if return_attn_map:
            return output * mask, attn_maps
        else:
            return output * mask