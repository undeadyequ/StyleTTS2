# coding:utf-8

import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional

from munch import Munch
import yaml
from torch.nn.utils.rnn import pad_sequence
from Modules.latent_diffusion.ddpm_dit import LatentDiffusion
from utils import get_phone_range_by_cut_f2p_attn, align_attention_to_zero


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError(
                'Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)


class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in,
                                           output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in,
                                           output_padding=1, padding=1)
        else:
            raise RuntimeError(
                'Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                'Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError(
                'Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384, repeat_num = 2):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)

        return s


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for l in self.main:
            x = l(x)
            features.append(x)
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features


class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p

        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == 'none':
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)

        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)

        x.masked_fill_(m, 0.0)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)

        #print("h_s mean, std, min, max: ", h.mean().item(), h.std().item(), h.min().item(),
        #      h.max().item())

        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        #print("gamma mean, std, min, max: ", gamma.mean().item(), gamma.std().item(), gamma.min().item(), gamma.max().item())
        #print("beta mean, variance, min, max", beta.mean().item(), beta.std().item(), beta.min().item(), beta.max().item())
        return (1 + gamma) * self.norm(x) + beta


class TemporalAdaIN1d(nn.Module):
    def __init__(self, style_dim, trend_dim, num_features, need_norm_trend=False):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        # Separate MLPs to decouple Speaker Identity from Local Trend
        self.mlp_s = nn.Linear(style_dim, num_features * 2)

        if need_norm_trend:
            self.mlp_t = spectral_norm(nn.Linear(trend_dim, num_features * 2, bias=False))
        else:
            self.mlp_t = nn.Linear(trend_dim, num_features * 2, bias=False)

        # Learnable scalar to control trend strength for robustness
        # Initialized small (0.1) to favor the stable global style initially
        self.alpha = nn.Parameter(torch.tensor(-2.197))  # sigmoid = 0.1

    def forward(self, x, s, T_i, trend_strength=1.0):
        # x: [B, num_features, T] (BERT features)
        # s: [B, style_dim] (Global Speaker/Style)
        # T_i: [B, trend_dim, T] (Phoneme-level trend)

        # 1. Global Speaker Baseline
        #print("s before mlp mean, variance, min, max", f"{s.mean().item():.3f}", f"{s.std().item():.3f}", f"{s.min().item():.3f}", f"{s.max().item():.3f}")
        h_s = self.mlp_s(s).unsqueeze(-1)  # [B, num_features*2, 1]
        #print("h_s after mlp mean, variance, min, max", f"{h_s.mean().item():.3f}", f"{h_s.std().item():.3f}", f"{h_s.min().item():.3f}", f"{h_s.max().item():.3f}")

        # 2. Local Trend Offset (Spatial Modulation)
        #print("T_i before mlp mean, variance, min, max", f"{T_i.mean().item():.3f}", f"{T_i.std().item():.3f}", f"{T_i.min().item():.3f}", f"{T_i.max().item():.3f}")

        h_t = self.mlp_t(T_i.transpose(1, 2)).transpose(1, 2)  # [B, num_features*2, T]
        #print("h_t before tanh mean, variance, min, max", f"{h_t.mean().item():.3f}", f"{h_t.std().item():.3f}", f"{h_t.min().item():.3f}", f"{h_t.max().item():.3f}")
        h_t = torch.tanh(h_t)
        #print("h_t after tanh mean, variance, min, max", f"{h_t.mean().item():.3f}", f"{h_t.std().item():.3f}", f"{h_t.min().item():.3f}", f"{h_t.max().item():.3f}")
        """
        print("h_s mean, std, min, max: ", h_s.mean().item(), h_s.std().item(), h_s.min().item(),
              h_s.max().item())
        print("h_t mean, variance, min, max", h_t.mean().item(), h_t.std().item(), h_t.min().item(),
              h_t.max().item())
        """
        if h_t.size(-1) * 2 == x.size(-1):
            h_t = F.interpolate(h_t, size=x.size(-1), mode='nearest')

        # 3. Combine with learnable alpha for mismatch protection
        if not self.training:
            h = h_s + (torch.sigmoid(self.alpha) * h_t * trend_strength)
        else:
            h = h_s + (torch.sigmoid(self.alpha) * h_t)

        ### Add Trend only on beta
        gamma_s, _ = torch.chunk(h_s, chunks=2, dim=1)
        _, beta = torch.chunk(h, chunks=2, dim=1)
        #gamma, beta = torch.chunk(h, chunks=2, dim=1)
        #print("gamma mean, std, min, max: ", gamma.mean().item(), gamma.std().item(), gamma.min().item(), gamma.max().item())
        #print("beta mean, variance, min, max", beta.mean().item(), beta.std().item(), beta.min().item(), beta.max().item())
        return (1 + gamma_s) * self.norm(x) + beta


class TemporalAdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, trend_dim, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0, need_norm_trend=False):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.need_norm_trend = need_norm_trend
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim, trend_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim, trend_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = TemporalAdaIN1d(style_dim, trend_dim, dim_in, self.need_norm_trend)
        self.norm2 = TemporalAdaIN1d(style_dim, trend_dim, dim_out, self.need_norm_trend)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, T_i):
        x = self.norm1(x, s, T_i)  # Pass T_i to TemporalAdaIN1d
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s, T_i)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s, T_i):
        out = self._residual(x, s, T_i)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()

        self.text_encoder = DurationEncoder(sty_dim=style_dim,
                                            d_model=d_hid,
                                            nlayers=nlayers,
                                            dropout=dropout)

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        batch_size = d.shape[0]
        text_size = d.shape[1]

        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)

        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)

        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))

        en = (d.transpose(-1, -2) @ alignment)

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))

        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class ProsodyPredictorByTrend(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, trd_dim, max_dur=50, dropout=0.1, trd_min=50, trd_max=600,
                 txt_trd_combine_type="concat"):
        super().__init__()

        self.text_encoder = DurationEncoder(sty_dim=style_dim,
                                            d_model=d_hid,
                                            nlayers=nlayers,
                                            dropout=dropout)
        self.txt_trd_combine_type = txt_trd_combine_type
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

        # Style fusion
        self.ln_text = nn.LayerNorm(d_hid + style_dim)
        self.emb_dim = 36
        if self.txt_trd_combine_type == "concat":
            self.n_bins = 8
            self.n_bins_interv = self.n_bins - 1
            self.ln_pitch = nn.LayerNorm(trd_dim)
            self.fuse_dim = d_hid + style_dim + trd_dim  # 512 + 128 + 36 = 676
        else:
            self.n_bins = 9  # add 1 for unvoiced
            self.n_bins_interv = self.n_bins - 2  # 1 is spared used for unvoiced
            self.trend_projection = nn.Linear(self.emb_dim, d_hid + style_dim)
            nn.init.zeros_(self.trend_projection.weight)
            nn.init.zeros_(self.trend_projection.bias)
            self.ln_pitch = nn.LayerNorm(d_hid + style_dim)
            self.fuse_dim = d_hid + style_dim
            self.trend_scale = nn.Parameter(torch.tensor(0.01)) # initialized to be very small

        self.shared = nn.LSTM(self.fuse_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)

        # trend operation: discretization, gaussian kernel, drop out
        self.pitch_bins = nn.Parameter(torch.linspace(torch.log(torch.tensor(trd_min)), torch.log(torch.tensor(trd_max)),
                                                      self.n_bins_interv), requires_grad=False)
        self.pitch_embedding = nn.Embedding(self.n_bins, self.emb_dim)
        self.null_trend_embed = nn.Parameter(torch.zeros(1, self.emb_dim, 1))
        with torch.no_grad():
            self.null_trend_embed.copy_(self.pitch_embedding.weight.mean(dim=0).unsqueeze(0).unsqueeze(-1))

        # trend operation: smooth conv and Gaussian kernel
        self.conv_layer = nn.Conv1d(self.emb_dim, self.emb_dim, 3, padding=1)
        self.kernel_size, self.sigma = 11, 2.0
        x_coord = torch.arange(self.kernel_size) - (self.kernel_size - 1) / 2
        gauss_kernel = torch.exp(-x_coord.pow(2) / (2 * self.sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()  # Normalize to sum to 1
        gauss_kernel = gauss_kernel.view(1, 1, -1).repeat(self.emb_dim, 1, 1)
        self.register_buffer('gaussian_kernel', gauss_kernel)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        batch_size = d.shape[0]
        text_size = d.shape[1]

        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)

        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)

        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))

        en = (d.transpose(-1, -2) @ alignment)

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, trd_log, s, uv_mask=None, drop_trend=False):
        if drop_trend:
            p_emb = self.null_trend_embed.expand(x.size(0), -1, x.size(-1))
        else:
            # 1. discretization, u/v mask, embedding, length alignment
            trd_log = trd_log[:, 0, :].clamp(self.pitch_bins[0].item(), self.pitch_bins[-1].item())
            trend_emb_idx = torch.bucketize(trd_log, self.pitch_bins) + 1  # to spare 0 to unvoiced
            trend_emb_idx = torch.clamp(trend_emb_idx, 1, self.n_bins - 1)
            if uv_mask is not None:
                trend_emb_idx = (trend_emb_idx * uv_mask).long()
            p_emb = self.pitch_embedding(trend_emb_idx).transpose(-1, -2)
            if p_emb.size(-1) != x.size(-1):
                p_emb = F.interpolate(p_emb, size=x.size(-1), mode="linear", align_corners=True)
            """
            with torch.no_grad():
                print("idx min/max:", trend_emb_idx.min().item(), trend_emb_idx.max().item())
                hist = torch.bincount(trend_emb_idx.flatten(), minlength=self.n_bins).float()
                print("bin usage:", (hist > 0).sum().item(), "/", self.n_bins)
                # should be within [0, n_bins-1]
            """

        # Smooth and gaussian kernel
        p_emb = self.conv_layer(p_emb)  # (B, d/2, T)
        #p_emb = F.conv1d(p_emb, self.gaussian_kernel, padding=5, groups=p_emb.size(1))
        #p_emb = F.avg_pool1d(p_emb, kernel_size=11, stride=1, padding=5)

        # LayerNorm x and trd, and add with gate (or concate)
        print("x mean, std, min, max: ", f"{x.mean().item():.3f}", f"{x.std().item():.3f}",
              f"{x.min().item():.3f}", f"{x.max().item():.3f}")
        print("p_emb mean, variance, min, max", f"{p_emb.mean().item():.3f}", f"{p_emb.std().item():.3f}",
              f"{p_emb.min().item():.3f}", f"{p_emb.max().item():.3f}")

        txt_emb = self.ln_text(x.transpose(1, 2)).transpose(1, 2)
        if self.txt_trd_combine_type == "concat":
            trd_emb = self.ln_pitch(p_emb.transpose(1, 2)).transpose(1, 2)
            x = torch.cat([txt_emb, trd_emb], dim=1)
        else:
            trd_emb = self.trend_projection(p_emb.transpose(1, 2))
            trd_emb = self.ln_pitch(trd_emb).transpose(1, 2)
            current_scale = torch.sigmoid(self.trend_scale)
            x = txt_emb +  current_scale * trd_emb * uv_mask.unsqueeze(1) # Clean, stable addition

        print("txt_emb mean, std, min, max: ", f"{txt_emb.mean().item():.3f}", f"{txt_emb.std().item():.3f}",
              f"{txt_emb.min().item():.3f}", f"{txt_emb.max().item():.3f}")
        print("trd_emb mean, variance, min, max", f"{trd_emb.mean().item():.3f}", f"{trd_emb.std().item():.3f}",
              f"{trd_emb.min().item():.3f}", f"{trd_emb.max().item():.3f}")
        print("s mean, variance, min, max", f"{s.mean().item():.3f}", f"{s.std().item():.3f}",
              f"{s.min().item():.3f}", f"{s.max().item():.3f}")

        # LSTM
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class ProsodyPredictorByTrend_v2(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, trend_dim, max_dur=50, dropout=0.1, trd_min=50, trd_max=600,
                 n_bins=9, need_norm_trend=False):
        """
        n_bins=9: # add 1 ("0th") bin for unvoiced
        """
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.need_norm_trend = need_norm_trend

        # F0/N prediction net
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(TemporalAdainResBlk1d(d_hid, d_hid, trend_dim, style_dim, dropout_p=dropout, need_norm_trend=need_norm_trend))
        self.F0.append(TemporalAdainResBlk1d(d_hid, d_hid // 2, trend_dim, style_dim, upsample=True, dropout_p=dropout, need_norm_trend=need_norm_trend))
        self.F0.append(TemporalAdainResBlk1d(d_hid // 2, d_hid // 2, trend_dim, style_dim, dropout_p=dropout, need_norm_trend=need_norm_trend))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

        # trend setting
        self.trend_dim = trend_dim
        self.n_bins = n_bins

        # trend discretizer, embedding, drop out null, smooth conv
        self.pitch_bins = nn.Parameter(torch.linspace(torch.log(torch.tensor(trd_min)), torch.log(torch.tensor(trd_max)),
                                                      self.n_bins - 2), requires_grad=False) # "0th" is spared used for unvoiced
        if self.need_norm_trend:
            self.trend_embedding = nn.Embedding(self.n_bins, trend_dim, max_norm=1.0)
            self.conv_layer = spectral_norm(nn.Conv1d(trend_dim, trend_dim, 3, padding=1))
        else:
            self.trend_embedding = nn.Embedding(self.n_bins, trend_dim)
            self.conv_layer = nn.Conv1d(trend_dim, trend_dim, 3, padding=1)

        self.null_trend_embed = nn.Parameter(torch.zeros(1, trend_dim, 1))
        with torch.no_grad():
            self.null_trend_embed.copy_(self.trend_embedding.weight.mean(dim=0).unsqueeze(0).unsqueeze(-1))


    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        batch_size = d.shape[0]
        text_size = d.shape[1]

        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)

        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)

        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))

        en = (d.transpose(-1, -2) @ alignment)

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, trd_log, s, uv_mask, drop_trend=False):
        """
        trd_log must be log!
        """
        if drop_trend:
            trend_emb = self.null_trend_embed.expand(x.size(0), -1, x.size(-1))
        else:
            # 1. cap, discretization, u/v mask, embedding, length alignment
            trd_log = trd_log[:, 0, :].clamp(self.pitch_bins[0].item(), self.pitch_bins[-1].item())
            trend_emb_idx = torch.bucketize(trd_log, self.pitch_bins) + 1  # to spare 0 to unvoiced
            trend_emb_idx = (trend_emb_idx * uv_mask).long()
            trend_emb = self.trend_embedding(trend_emb_idx).transpose(-1, -2)
            if trend_emb.size(-1) != x.size(-1):
                trend_emb = F.interpolate(trend_emb, size=x.size(-1), mode="linear", align_corners=True)
            """CHECK Bin Usage
            with torch.no_grad():
                print("idx min/max:", trend_emb_idx.min().item(), trend_emb_idx.max().item())
                hist = torch.bincount(trend_emb_idx.flatten(), minlength=self.n_bins).float()
                print("bin usage:", (hist > 0).sum().item(), "/", self.n_bins)
                # should be within [0, n_bins-1]
            """
        # Smooth and gaussian kernel
        trend_emb = self.conv_layer(trend_emb)  # (B, d/2, T)
        #print(trend_emb[0, 0, :20])
        trend_emb = trend_emb * uv_mask.unsqueeze(1)
        #print(trend_emb[0,0,:20])
        #print(uv_mask[0,:20])
        #trend_emb = F.conv1d(trend_emb, self.gaussian_kernel, padding=5, groups=trend_emb.size(1))
        #trend_emb = F.avg_pool1d(trend_emb, kernel_size=11, stride=1, padding=5)

        # LSTM
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s, trend_emb)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class ProsodyPredictorByTrend_v3(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, trd_dim, max_dur=50, dropout=0.1, trd_min=50, trd_max=600,
                 n_bins=8):
        super().__init__()
        self.dropout = dropout
        self.n_bins = n_bins
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

        # trend operation: discretization, emb, conv, layer_norm, drop out
        self.trd_bins = nn.Parameter(torch.linspace(torch.log(torch.tensor(trd_min)), torch.log(torch.tensor(trd_max)), n_bins - 1), requires_grad=False)
        self.trd_embed = nn.Embedding(n_bins, trd_dim, max_norm=1.0)
        self.trd_conv = spectral_norm(nn.Conv1d(trd_dim, trd_dim, 3, padding=1))
        self.trd_ln = nn.LayerNorm(trd_dim, elementwise_affine=False)
        self.trd_null = nn.Parameter(torch.zeros(1, trd_dim, 1))
        with torch.no_grad():
            self.trd_null.copy_(self.trd_embed.weight.mean(dim=0).unsqueeze(0).unsqueeze(-1))

        # Style fusion
        self.fuse_dim = d_hid + style_dim + trd_dim  # 512 + 128 + 4 = 644
        self.shared = nn.LSTM(self.fuse_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        batch_size = d.shape[0]
        text_size = d.shape[1]

        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)

        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)

        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))

        # Concatenate d with pl_trend
        en = (d.transpose(-1, -2) @ alignment)  # frame-level prosodic embedding

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, trd_log, s, uv_mask=None, drop_trend=False):
        if drop_trend:
            p_emb = self.trd_null.expand(x.size(0), -1, x.size(-1))
        else:
            # 1. discretization, embedding, length alignment
            trd_log = trd_log[:, 0, :].clamp(self.trd_bins[0].item(), self.trd_bins[-1].item())
            trend_emb_idx = torch.bucketize(trd_log, self.trd_bins)  # to spare 0 to unvoiced
            trend_emb_idx = torch.clamp(trend_emb_idx, 0, self.n_bins - 1)
            #print("trd_log", trd_log)
            #print("trend_emb_idx", trend_emb_idx)
            p_emb = self.trd_embed(trend_emb_idx).transpose(-1, -2)
            if p_emb.size(-1) != x.size(-1):
                p_emb = F.interpolate(p_emb, size=x.size(-1), mode="linear", align_corners=True)
            """Check bins
            with torch.no_grad():
                print("idx min/max:", trend_emb_idx.min().item(), trend_emb_idx.max().item())
                hist = torch.bincount(trend_emb_idx.flatten(), minlength=self.n_bins).float()
                print("bin usage:", (hist > 0).sum().item(), "/", self.n_bins)
                # should be within [0, n_bins-1]
            """
        p_emb = self.trd_conv(p_emb)  # (B, d/2, T)

        """check x, p_emb variance
        # LayerNorm x and trd, and add with gate (or concate)
        print("x mean, std, min, max: ", f"{x.mean().item():.3f}", f"{x.std().item():.3f}",
              f"{x.min().item():.3f}", f"{x.max().item():.3f}")
        print("trd_emb (before ln) mean, variance, min, max", f"{p_emb.mean().item():.3f}", f"{p_emb.std().item():.3f}",
              f"{p_emb.min().item():.3f}", f"{p_emb.max().item():.3f}")
        """
        trd_emb = self.trd_ln(p_emb.transpose(1, 2)).transpose(1, 2)
        trd_emb = F.dropout(trd_emb, p=self.dropout, training=self.training)
        x = torch.cat([x, trd_emb], dim=1)

        """
        print("trd_emb mean (after ln), variance, min, max", f"{trd_emb.mean().item():.3f}", f"{trd_emb.std().item():.3f}",
              f"{trd_emb.min().item():.3f}", f"{trd_emb.max().item():.3f}")
        print("s mean, variance, min, max", f"{s.mean().item():.3f}", f"{s.std().item():.3f}",
              f"{s.min().item():.3f}", f"{s.max().item():.3f}")
        """

        # LSTM
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

class ProsodyPredictorByTrend_v4(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, trd_dim, max_dur=50, dropout=0.1, trd_min=50, trd_max=600, n_bins=8):
        super().__init__()
        self.dropout = dropout
        self.n_bins = n_bins
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

        # Encode pl_trend
        self.trd_encoder = TrendEncoder(n_bins=n_bins, trd_dim=trd_dim, trd_min=trd_min, trd_max=trd_max, dropout=dropout)

        # Style fusion
        self.fuse_dim = d_hid + style_dim + trd_dim  # 512 + 128 + 4 = 644
        self.shared = nn.LSTM(self.fuse_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        batch_size = d.shape[0]
        text_size = d.shape[1]

        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(d, input_lengths, batch_first=True, enforce_sorted=False)

        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)

        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))  # why 0.5?

        # Concatenate d with pl_trend
        en = (d.transpose(-1, -2) @ alignment)  # frame-level prosodic embedding

        return duration.squeeze(-1), en

    def trd_encoding(self, f0_gd_frame, f2p_attn_gd, uv_mask_phn_gd, uv_mask_phn_tgt=None, f2p_attn_pred=False, tgt_lengths_phn=None,
                     drop_trend=False, eps: float=1e-8, return_frame_level=True, return_trd_index=False):
        """
        Encode fusion of phoneme-level bert and trend signal into duplicated frame-level bert_trend
        1) phoneme averaging
        2) uv_mask_phn_gd masking
        3) Re-mask process: interp unvoiced of f0_gd_phn, upsample to tgt_len, and re-uv_mask to ref_len (ONLY in inference)
        4) trd encoder
        5) bert trd fusion

        f0_gd_frame [B, Tf]
        f2p_attn_gd [B, Tp, Tf/2]: phoneme duration attention for given reference speech (half frame in attn because it is asr attn)
        uv_mask_phn_gd [B, Tp]

        # ONLY in inference:
        f2p_attn_pred [B, Tp_tgt, Tf_tgt/2]: phoneme duration attention for target speech by duration predictor
        uv_mask_phn_tgt [B,Tp_tgt]: uv mask of target phoneme for re-uvmask

        return
        trd_fl: [B, trd_dim, Tf]
        """
        attn_f2p = F.interpolate(f2p_attn_gd, size=f0_gd_frame.size(-1), mode="nearest")  # upsample asr-based f2p attn to match f0_gd_frame

        # phoneme averaging
        w = attn_f2p.transpose(1, 2).to(f0_gd_frame.dtype)  # [B, Tf, Tp]
        denom = w.sum(dim=1).clamp_min(1.0)  # [B, Tp]
        f0_gd_phn = torch.einsum("bt,btp->bp", f0_gd_frame, w) / (denom + eps)  # [B, Tp]

        # 2. Apply ground-truth masking
        """SANITY CHECK"""
        #print("f0_gd_phn before mask", f0_gd_phn[0, :])
        f0_gd_phn = f0_gd_phn * uv_mask_phn_gd
        #print("f0_gd_phn after mask", f0_gd_phn[0, :])

        # 3. Inference Logic: Sample-wise Slicing and Alignment (B >= 1)
        if not self.training:
            if uv_mask_phn_tgt is None or f2p_attn_pred is None:
                raise ValueError("Inference requires uv_mask_phn_tgt and f2p_attn_pred!")

            # Inferred target lengths from attention mask if not provided
            if tgt_lengths_phn is None:
                tgt_lengths_phn = (f2p_attn_pred.sum(dim=-1) > 0).sum(dim=-1)

            #f2p_attn_pred_upaligned = torch.zeros(f2p_attn_pred.size(0), torch.max(tgt_lengths_phn), f2p_attn_pred.size(1)).to(f0_gd_frame.device)
            # Step A: Find spoken boundaries in reference
            _, _, phn_start, phn_end = get_phone_range_by_cut_f2p_attn(f2p_attn_gd)

            f0_final_sample_list = []
            for i in range(f0_gd_phn.size(0)):
                # Step B: Slice Reference to remove padding
                s, e = phn_start[i], phn_end[i] + 1
                f0_gd_phn_clean = f0_gd_phn[i:i + 1, s:e]  # [1, Tp_actual]
                uv_gd_phn_clean = uv_mask_phn_gd[i:i + 1, s:e]  # [1, Tp_actual]

                # Step C: Interpolate gaps (Isolated from padding)
                f0_gd_phn_interpU = _interp_unvoiced_1d(f0_gd_phn_clean, uv_gd_phn_clean.bool())

                # Step D: Rescale to sample-specific target length
                tgt_phn_len = int(tgt_lengths_phn[i])
                f0_rescaled = F.interpolate(
                    f0_gd_phn_interpU.unsqueeze(1),
                    size=tgt_phn_len,
                    mode="linear",
                    align_corners=True
                ).squeeze(1)  # [1, tgt_phn_len]

                # Step E: Apply Target Mask
                f0_final_sample = f0_rescaled * uv_mask_phn_tgt[i:i + 1, :tgt_phn_len]
                f0_final_sample_list.append(f0_final_sample.squeeze(0))

            # Re-pack into batch [B, max_Tp_tgt]
            f0_gd_phn = pad_sequence(f0_final_sample_list, batch_first=True)

        # 4. Trend Encoder
        if return_trd_index:
            trd_pl, trd_pl_index = self.trd_encoder(f0_gd_phn.unsqueeze(1), drop_trend=drop_trend, return_trd_index=return_trd_index)
        else:
            trd_pl  = self.trd_encoder(f0_gd_phn.unsqueeze(1), drop_trend=drop_trend, return_trd_index=return_trd_index)

        # 5. Broadcast back to frames
        if return_frame_level:
            if self.training:
                trd_fl = (trd_pl @ f2p_attn_gd)
            else:
                if trd_pl.size(-1) != f2p_attn_pred.size(-2):
                    if trd_pl.size(0) > 1:
                        f2p_attn_pred = align_attention_to_zero(f2p_attn_pred)
                    #if trd_pl.size(-1) == f2p_attn_pred.size(-2) + 1:  # uv_mask (same size with trd_pl) appended 0 at start point.
                    #    trd_pl = trd_pl[:, :, 1:]
                    #else:
                trd_fl = (trd_pl @ f2p_attn_pred)
            if return_trd_index:
                return trd_fl, (trd_pl_index.to(f2p_attn_pred.dtype) @ f2p_attn_pred).long()
            return trd_fl
        return trd_pl

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class TrendEncoder(nn.Module):
    def __init__(self, n_bins=32, trd_dim=4, trd_min=50, trd_max=600, dropout=0.2):
        """Discretize and embed trend signal"""
        super().__init__()
        self.trd_bins = nn.Parameter(torch.linspace(torch.tensor(trd_min), torch.tensor(trd_max), n_bins - 1), requires_grad=False)
        self.trd_embed = nn.Embedding(n_bins, trd_dim, max_norm=1.0)
        self.trd_ln = nn.LayerNorm(trd_dim, elementwise_affine=False)
        self.trd_null = nn.Parameter(torch.zeros(1, trd_dim, 1))
        self.drop_out = dropout
        self.trd_min = trd_min
        self.trd_max = trd_max
        self.n_bins = n_bins
        with torch.no_grad():
            self.trd_null.copy_(self.trd_embed.weight.mean(dim=0).unsqueeze(0).unsqueeze(-1))

    def forward(self, trd_pl, drop_trend=False, return_trd_index=False):
        """
        args:
            trd_pl: (B, 1, T_p)
        return
            trd_emb:(B, trd_dim, T_p)
        """
        if drop_trend:
            p_emb = self.trd_null.expand(trd_pl.size(0), -1, trd_pl.size(-1))
            trend_emb_idx = torch.zeros_like(p_emb).to(p_emb.device)
        else:
            # 1. discretization, embedding, length alignment
            trd = trd_pl[:, 0, :].clamp(self.trd_bins[0].item(), self.trd_bins[-1].item())
            trend_emb_idx = torch.bucketize(trd, self.trd_bins)  # to spare 0 to unvoiced
            """SANITY CHECK"""
            #print("trd, ", trd[0, :])
            #print("trend_em_idx", trend_emb_idx[0, ...])
            p_emb = self.trd_embed(trend_emb_idx).transpose(-1, -2)
        trd_emb = self.trd_ln(p_emb.transpose(1, 2)).transpose(1, 2)
        trd_emb = F.dropout(trd_emb, p=self.drop_out, training=self.training)

        if return_trd_index:
            return trd_emb, trend_emb_idx
        return trd_emb

class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim,
                                      d_model // 2,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=True,
                                      dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))

        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)

        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)

        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)

                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)

        return x.transpose(-1, -2)

    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


def load_F0_models(path):
    # load F0 model
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()

    return F0_model


def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu', weights_only=False)['model']
        model.load_state_dict(params)
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model


def build_model(args, args_cfm, text_aligner, pitch_extractor, bert):
    # choose different decoder version
    if "pe_mu_type" in args and args.pe_mu_type == "down_pe_up_mel":
        from Modules.cfm.flow_matching_down_pe import CFMDecoder
        decoder = CFMDecoder(**args_cfm)
    elif  "pe_mu_type" in args and args.pe_mu_type == "up_mu_v2":
        from Modules.cfm.flow_matching2 import CFMDecoder
        decoder = CFMDecoder(**args_cfm)
    elif  "cond_prosody_type" in args and args.cond_prosody_type == "hierstyle":
        from Modules.cfm.flow_matching_hier import CFMDecoder
        decoder = CFMDecoder(**args_cfm)
    else:
        from Modules.cfm.flow_matching import CFMDecoder
        decoder = CFMDecoder(**args_cfm)

    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)

    if "cond_prosody_type" in args:
        if args.cond_prosody_type == "simplefuse":
            txt_trd_combine_type = args.get("txt_trd_combine_type", "concat")
            predictor = ProsodyPredictorByTrend(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer,
                                                trd_dim=args.trd_dim, max_dur=args.max_dur, dropout=args.dropout,
                                                txt_trd_combine_type=txt_trd_combine_type)
        elif "temporalAdaIN" in args.cond_prosody_type:
            need_norm_trend = True if "_norm" in args.cond_prosody_type else False
            predictor = ProsodyPredictorByTrend_v2(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer,
                                                trend_dim=args.trd_dim, max_dur=args.max_dur, dropout=args.dropout, need_norm_trend=need_norm_trend)
        elif args.cond_prosody_type == "simplefuse_v3":
            predictor = ProsodyPredictorByTrend_v3(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer,
                                                trd_dim=args.trd_dim, max_dur=args.max_dur, dropout=args.dropout, n_bins=args.n_bins)
        elif args.cond_prosody_type == "bertfusion":
            predictor = ProsodyPredictorByTrend_v4(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer,
                                                trd_dim=args.trd_dim, max_dur=args.max_dur, dropout=args.dropout, n_bins=args.n_bins,
                                                   trd_min=args.trd_min, trd_max=args.trd_max)
    else:
        predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer,
                                     max_dur=args.max_dur, dropout=args.dropout)
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim,
                                 max_conv_dim=args.hidden_dim, repeat_num=args.repeat_num)  # acoustic style encoder
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim,
                                     max_conv_dim=args.hidden_dim)  # prosodic style encoder
    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim * 2,
                                         context_embedding_features=bert.config.hidden_size,
                                         context_features=args.style_dim * 2,
                                         **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim * 2,
                                    context_embedding_features=bert.config.hidden_size,
                                    **args.diffusion.transformer)
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba,  # Conditional dropout of batch elements,
        channels=args.style_dim * 2,
        context_features=args.style_dim * 2,
    )
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean=args.diffusion.dist.mean, std=args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data,
        # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    discriminator = Discriminator2d(dim_in=args.dim_in, num_domains=1, max_conv_dim=args.hidden_dim)

    nets = Munch(
        bert=bert,
        bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

        predictor=predictor,
        decoder=decoder,
        text_encoder=text_encoder,

        predictor_encoder=predictor_encoder,
        style_encoder=style_encoder,
        diffusion=diffusion,

        text_aligner=text_aligner,
        pitch_extractor=pitch_extractor,

        discriminator=discriminator
    )
    return nets


def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key], strict=False)
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0
    return model, optimizer, epoch, iters


@torch.no_grad()
def _interp_unvoiced_1d(y: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
    """
    Fill unvoiced positions in y by linear interpolation between nearest voiced samples.
    y:      [B, T] float
    voiced: [B, T] bool
    """
    B, T = y.shape
    device = y.device
    idx = torch.arange(T, device=device)

    y_filled = y.clone()
    for b in range(B):
        vb = voiced[b]
        if vb.any():
            v_idx = idx[vb]
            v_val = y[b, vb]

            r = torch.searchsorted(v_idx, idx).clamp(0, v_idx.numel() - 1)
            l = (r - 1).clamp(0, v_idx.numel() - 1)

            x0, x1 = v_idx[l].float(), v_idx[r].float()
            y0, y1 = v_val[l], v_val[r]

            denom = (x1 - x0).clamp_min(1.0)  # avoid 0 when l==r
            w = ((idx.float() - x0) / denom).clamp(0.0, 1.0)
            y_filled[b] = y0 + (y1 - y0) * w
        else:
            # No voiced frames: nothing to interpolate from
            y_filled[b].zero_()
    return y_filled

if __name__ == '__main__':
    y = torch.tensor([0, 0, 0, 0, 6, 0, 5, 3, 0, 0, 0])
    voiced = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]).to(torch.bool)
    y_fill = _interp_unvoiced_1d(y.unsqueeze(0), voiced.unsqueeze(0))
    print(y_fill)