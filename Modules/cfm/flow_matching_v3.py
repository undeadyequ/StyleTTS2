"""Fixed CFM decoder (v3).

Drop-in replacement for flow_matching.py that uses DecoderV3 (estimator_v3)
instead of Decoder (estimator).  All other logic is identical.

Select via cfm_config in the YAML:
    cfm_config:
        use_v3: true   # use this file; false (default) → flow_matching.py
        cross_attn: true
        ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchdiffeq import odeint

from Modules.ditmodules.estimator_v3 import DecoderV3
from Modules.latent_diffusion.pre_dit_modules import ResBlk1d
from torch.nn.utils import weight_norm
from utilities.guide_mask import make_guided_attention_masks2
from Modules.prosodymodules.prosody_fusion import FactorizedGateMoEProsodyFusion, moeConfig


class CFMDecoderV3(torch.nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels,
                 cross_attn, residual_dim=64, dim_in=512, cfg_dropout=0,
                 prosody_fusion=False, glb_t_concate=False, use_lsc=True):
        super().__init__()
        self.noise_channels  = noise_channels
        self.cond_channels   = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.filter_channels = filter_channels
        self.gin_channels    = gin_channels
        self.sigma_min       = 1e-4
        self.cross_attn      = cross_attn
        self.prosody_fusion  = prosody_fusion

        self.estimator = DecoderV3(
            noise_channels, cond_channels, hidden_channels, out_channels,
            filter_channels, p_dropout, n_layers, n_heads, kernel_size, gin_channels,
            cross_attn=cross_attn, glb_t_concate=glb_t_concate, use_lsc=use_lsc)

        pe_emb_dim  = 256
        asr_res_dim = 80

        self.pe_encode = nn.Sequential(
            ResBlk1d(2, residual_dim, normalize=True),
            ResBlk1d(residual_dim, pe_emb_dim, normalize=True))

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, asr_res_dim, kernel_size=3, padding=1)),
            nn.InstanceNorm1d(asr_res_dim, affine=True))

        self.cfg_dropout = cfg_dropout
        if self.cfg_dropout > 0:
            self.fake_glb     = nn.Parameter(torch.zeros(1, gin_channels))
            self.fake_content = nn.Parameter(torch.zeros(1, asr_res_dim, 1))
            self.fake_pe      = nn.Parameter(torch.zeros(1, pe_emb_dim, 1))

        self.attn_cache = list()
        self.t_count    = list()

    @torch.no_grad()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None, seq_style=None,
                p_mask=None, solver=None, cfg_strength=None, mono_guide_delta=-1.0,
                return_attn_map=True, seq_style_gt=None):
        mu        = F.interpolate(mu, scale_factor=2, mode="nearest")
        mu        = self.asr_res(mu)
        seq_style = self.pe_encode(seq_style)

        if seq_style_gt is not None and self.prosody_fusion:
            seq_style_gt = self.pe_encode(seq_style_gt)
            seq_style, piRef_e_a_gap = self.prosody_fuser(
                mu.transpose(1, 2), c,
                seq_style.transpose(1, 2), seq_style_gt.transpose(1, 2),
                return_gates=True)
            seq_style = seq_style.transpose(1, 2)

        z      = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        mask   = torch.ones([z.size(0), 1, z.size(-1)]).to(z.device)
        p_mask = torch.ones([seq_style.size(0), 1, seq_style.size(-1)]).to(z.device)
        ilens  = [z[i].size(-1) for i in range(z.size(0))]
        olens  = [seq_style[i].size(-1) for i in range(seq_style.size(0))]

        if mono_guide_delta > 0:
            regularize_attn_map = make_guided_attention_masks2(
                ilens, olens, base_sigma=mono_guide_delta, eps=0.002)
            regularize_attn_map = 1 - regularize_attn_map
        else:
            regularize_attn_map = None

        if cfg_strength is None:
            estimator = functools.partial(
                self.estimator, mask=mask, mu=mu, c=c, seq_style=seq_style,
                p_mask=p_mask, return_attn_map=return_attn_map,
                regularize_attn_map=regularize_attn_map)
        else:
            if self.cfg_dropout <= 0:
                raise IOError("cfg_dropout must be > 0 during training for CFG inference")
            estimator = functools.partial(
                self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_strength=cfg_strength,
                seq_style=seq_style, p_mask=p_mask, return_attn_map=return_attn_map,
                regularize_attn_map=regularize_attn_map)

        trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)

        if len(self.attn_cache) != 0:
            attn_maps = torch.stack(self.attn_cache).clone()
            self.t_count, self.attn_cache = list(), list()
        else:
            attn_maps = None

        if self.prosody_fusion:
            return trajectory[-1], attn_maps, piRef_e_a_gap
        return trajectory[-1], attn_maps

    def cfg_wrapper(self, t, x, mask, mu, c, cfg_strength, seq_style=None, p_mask=None,
                    return_attn_map=False, regularize_attn_map=None):
        fake_glb     = self.fake_glb.repeat(x.size(0), 1)
        fake_content = self.fake_content.repeat(x.size(0), 1, x.size(-1))
        fake_pe      = self.fake_pe.repeat(seq_style.size(0), 1, seq_style.size(-1))

        cond_output   = self.estimator(t, x, mask, mu, c, seq_style, p_mask,
                                       return_attn_map, regularize_attn_map=regularize_attn_map)
        uncond_output = self.estimator(t, x, mask, fake_content, fake_glb, fake_pe,
                                       p_mask, return_attn_map)

        if return_attn_map and cond_output[1] is not None:
            self.t_count.append(t)
            if len(self.t_count) == 1 or len(self.t_count) == 100 or len(self.t_count) == 200:
                self.attn_cache.append(cond_output[1].squeeze())

        return uncond_output[0] + cfg_strength * (cond_output[0] - uncond_output[0])

    def compute_loss(self, x1, mask, mu, c, seq_style=None, p_mask=None,
                     regularize_attn_map=None, seq_style_gt=None):
        mu        = F.interpolate(mu, scale_factor=2, mode="nearest")
        mu        = self.asr_res(mu)
        seq_style = self.pe_encode(seq_style)

        if seq_style_gt is not None and self.prosody_fusion:
            seq_style_gt = self.pe_encode(seq_style_gt)
            seq_style, piRef_e_a_gap = self.prosody_fuser(
                mu.transpose(1, 2), c,
                seq_style.transpose(1, 2), seq_style_gt.transpose(1, 2),
                return_gates=True)
            seq_style = seq_style.transpose(1, 2)
        else:
            piRef_e_a_gap = None

        b, _, t = mu.shape

        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        t = 1 - torch.cos(t * 0.5 * torch.pi)

        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        x_mask = torch.ones([x1.size(0), 1, x1.size(-1)]).to(x1.device)
        p_mask = x_mask

        if self.cfg_dropout > 0:
            cfg_mask  = torch.rand(x1.size(0), 1, device=x1.device) > self.cfg_dropout
            c         = c * cfg_mask + ~cfg_mask * self.fake_glb.repeat(z.size(0), 1)
            cfg_mask  = cfg_mask.unsqueeze(-1)
            mu        = mu * cfg_mask + ~cfg_mask * self.fake_content.repeat(mu.size(0), 1, mu.size(-1))
            seq_style = seq_style * cfg_mask + ~cfg_mask * self.fake_pe.repeat(seq_style.size(0), 1, seq_style.size(-1))

        estm_out = self.estimator(t.squeeze(), y, x_mask, mu, c, seq_style, p_mask,
                                  regularize_attn_map=regularize_attn_map)
        loss = F.mse_loss(estm_out, u, reduction="sum") / (torch.sum(x_mask) * u.size(1))

        if self.prosody_fusion:
            return loss, None, piRef_e_a_gap
        return loss, None
