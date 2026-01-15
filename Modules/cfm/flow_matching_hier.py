import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchdiffeq import odeint
from utilities.vis import save_plot

from Modules.ditmodules.estimator import Decoder
from Modules.latent_diffusion.pre_dit_modules import ResBlk1d
from torch.nn.utils import weight_norm
from utilities.guide_mask import make_guided_attention_masks2
from Modules.prosodymodules.prosody_fusion import FactorizedGateMoEProsodyFusion, moeConfig
# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py

class CFMDecoder(torch.nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout, gin_channels, cross_attn, residual_dim=64, dim_in=512, cfg_dropout=0,
                 official_dit=False, prosody_fusion=True, pe_min_max=(100, 200 , 1, 3)):  # different DiT version
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 1e-4
        self.cross_attn = cross_attn
        self.prosody_fusion = prosody_fusion

        self.estimator = Decoder(noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, p_dropout, n_layers,
                                 n_heads, kernel_size, gin_channels, cross_attn=cross_attn, official_dit=official_dit)  # cond: mu, gin: global_style

        # adpative to styleTTS input (pitch/energy)

        #pitch_bin_nums, pitch_emb_dim = 40, 256
        pe_emb_dim = 256
        asr_res_dim = 80  # to concat with z
        #self.pitch_embed = torch.nn.Linear(pitch_bin_nums, pitch_emb_dim)

        self.pe_encode = nn.Sequential(ResBlk1d(2, residual_dim, normalize=True),
                                    ResBlk1d(residual_dim, pe_emb_dim, normalize=True))
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, asr_res_dim, kernel_size=3, padding=1)),  # keep same length of output
            nn.InstanceNorm1d(asr_res_dim, affine=True)
        )

        # add cfg
        self.cfg_dropout = cfg_dropout
        if self.cfg_dropout > 0:
            self.fake_glb = nn.Parameter(torch.zeros(1, gin_channels))
            self.fake_content = nn.Parameter(torch.zeros(1, asr_res_dim, 1))
            self.fake_pe = nn.Parameter(torch.zeros(1, pe_emb_dim, 1))

        self.attn_cache = list()
        self.t_count = list()

        if self.prosody_fusion:
            self.n_bins = 24
            self.emb_dim = 128
            self.pe_min_max = tuple(pe_min_max)
            self.pitch_bins = nn.Parameter(
                torch.linspace(self.pe_min_max[0], self.pe_min_max[1], self.n_bins - 1),
                requires_grad=False,
            )
            self.energy_bins = nn.Parameter(
                torch.linspace(self.pe_min_max[2], self.pe_min_max[3], self.n_bins - 1),
                requires_grad=False,
            )
            self.energy_embedding = nn.Embedding(self.n_bins, self.emb_dim)
            self.pitch_embedding = nn.Embedding(self.n_bins, self.emb_dim)
            self.conv_layer = nn.Conv1d(self.emb_dim, 128, 3, padding=1)
            self.pe_phone_encode = nn.Sequential(ResBlk1d(self.emb_dim, residual_dim, normalize=True),
                                                 ResBlk1d(residual_dim, pe_emb_dim, normalize=True))
            self.proj = nn.Linear(2 * pe_emb_dim, pe_emb_dim)
            self.gd_gate = nn.Parameter(torch.tensor(-6.0))  # sigmoid ~ 0.002 at start
            self.VIS_PITCH_REAL_DISCRETE = True

    @torch.no_grad()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None, seq_style=None, p_mask=None,
                solver=None, cfg_strength=None, mono_guide_delta=0, return_attn_map=True, seq_style_gt=None):
        """Forward diffusion
        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            c (torch.Tensor, optional): speaker embedding
                shape: (batch_size, gin_channels)
            solver: see https://github.com/rtqichen/torchdiffeq for supported solvers
            cfg_strength: used for cfg inference

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        # preprocess mu, seq_style, and t_span
        mu = F.interpolate(mu, scale_factor=2, mode="nearest")
        mu = self.asr_res(mu)

        # prosody fusion
        if seq_style_gt is not None and self.prosody_fusion:
            seq_style = self.fuse_pe_pred_gd(seq_style, seq_style_gt)
        else:
            seq_style = self.pe_encode(seq_style)  # (B, d, T)  d =256

        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        # create mask
        mask = torch.ones([z.size(0), 1, z.size(-1)]).to(z.device)   # currently mask not worked
        p_mask = torch.ones([seq_style.size(0), 1, seq_style.size(-1)]).to(z.device) # currently mask not worked
        ilens = [z[i].size(-1) for i in range(z.size(0))]
        olens = [seq_style[i].size(-1) for i in range(seq_style.size(0))]

        # create regularize_attn_map
        regularize_attn_map = make_guided_attention_masks2(ilens, olens, base_sigma=mono_guide_delta, eps=0.002)  # diagonal:0, other:->1
        regularize_attn_map = 1 - regularize_attn_map
        print_mono_guide_delta = str(mono_guide_delta).replace(".", "").replace("-", "m")
        save_plot(regularize_attn_map[0].detach().cpu(), f"monoMask_guassion_{print_mono_guide_delta}.png")

        # cfg control
        if cfg_strength is None:
            ### TODO-S: tempt code for returning attn_map of dit at t=0
            #_, attn_maps = self.estimator(t_span[0], z, mask=mask, mu=mu, c=c, seq_style=seq_style,
            #                              p_mask=p_mask, return_attn_map=True, regularize_attn_map=regularize_attn_map)  # for attn_maps
            estimator = functools.partial(self.estimator, mask=mask, mu=mu, c=c, seq_style=seq_style,
                                          p_mask=p_mask, return_attn_map=return_attn_map, regularize_attn_map=regularize_attn_map) # for trajectory
        else:
            if self.cfg_dropout <= 0:
                raise IOError("cfg_dropout should bigger than 0 in training if you want cfg in inference!!!")

            #_, attn_maps = self.cfg_wrapper(t_span[0], z, mask=mask, mu=mu, c=c, cfg_strength=cfg_strength, seq_style=seq_style,
            #                                p_mask=p_mask, return_attn_map=True, regularize_attn_map=regularize_attn_map)
            estimator = functools.partial(self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_strength=cfg_strength, seq_style=seq_style,
                                          p_mask=p_mask, return_attn_map=return_attn_map, regularize_attn_map=regularize_attn_map)
        ### TODO-B
        trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)

        # clear t_count and save attn_maps
        if len(self.attn_cache) != 0:
            attn_maps = torch.stack(self.attn_cache).clone()
            self.t_count, self.attn_cache = list(), list()
        else:
            attn_maps = None
        return trajectory[-1], attn_maps

    # cfg inference
    def cfg_wrapper(self, t, x, mask, mu, c, cfg_strength, seq_style=None, p_mask=None, return_attn_map=False, regularize_attn_map=None):
        fake_glb = self.fake_glb.repeat(x.size(0), 1)
        fake_content = self.fake_content.repeat(x.size(0), 1, x.size(-1))
        fake_pe = self.fake_pe.repeat(seq_style.size(0), 1, seq_style.size(-1))

        cond_output = self.estimator(t, x, mask, mu, c, seq_style, p_mask, return_attn_map, regularize_attn_map=regularize_attn_map)
        uncond_output = self.estimator(t, x, mask, fake_content, fake_glb, fake_pe, p_mask, return_attn_map)
        #uncond_output = self.estimator(t, x, mask, fake_content, fake_speaker, None, None, return_attn_map)

        if return_attn_map:  # Only return attn_map
            self.t_count.append(t)
            if len(self.t_count) == 1 or len(self.t_count) == 100 or len(self.t_count) == 200:  # attention save conditions
                self.attn_cache.append(cond_output[1].squeeze())
            output = uncond_output[0] + cfg_strength * (cond_output[0] - uncond_output[0])
            return output
        else:
            output = uncond_output + cfg_strength * (cond_output - uncond_output)
            return output

    def compute_loss(self, x1, mask, mu, c, seq_style=None, p_mask=None, regularize_attn_map=None,
                     seq_style_gt=None, pe_stats=None):
        """Computes diffusion loss
        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.
        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        mu = F.interpolate(mu, scale_factor=2, mode="nearest")
        mu = self.asr_res(mu)

        # encoder/merge phoneme-level pe: discrete -> emb -> interpolate, smooth, concate, and mlp
        if seq_style_gt is not None and self.prosody_fusion:
            seq_style = self.fuse_pe_pred_gd(seq_style, seq_style_gt)
        else:
            seq_style = self.pe_encode(seq_style)  # (B, d, T)  d =256

        # perturb mel given random t
        b, _, t = mu.shape
        # use cosine timestep scheduler from cosyvoice: https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        ############ TEMP code
        x_mask = torch.ones([x1.size(0), 1, x1.size(-1)]).to(x1.device)
        p_mask = x_mask

        # add cfg mask
        if self.cfg_dropout > 0:
            cfg_mask = torch.rand(x1.size(0), 1, device=x1.device) > self.cfg_dropout
            c = c * cfg_mask + ~cfg_mask * self.fake_glb.repeat(z.size(0), 1)
            cfg_mask = cfg_mask.unsqueeze(-1)
            mu = mu * cfg_mask + ~cfg_mask * self.fake_content.repeat(mu.size(0), 1, mu.size(-1))
            seq_style = seq_style * cfg_mask + ~cfg_mask * self.fake_pe.repeat(seq_style.size(0), 1, seq_style.size(-1))

        estm_out, attn_maps = self.estimator(t.squeeze(), y, x_mask, mu, c, seq_style, p_mask, regularize_attn_map=regularize_attn_map)
        loss = F.mse_loss(estm_out, u, reduction="sum") / (torch.sum(x_mask) * u.size(1))
        #return loss, y
        #return loss, estm_out, attn_maps
        return loss, attn_maps

    def fuse_pe_pred_gd(self, seq_style, seq_style_gt):
        """
        gt: discrete -> emb -> interp -> smooth -> concate -> pe_encoder -> proj
        """
        if False:
            e_emb = torch.bucketize(seq_style_gt[:, 0, :], self.energy_bins)
            #print("e_emb q25, q75", torch.quantile(e_emb[0].float(), q=0.25).item(), torch.quantile(e_emb[0].float(), q=0.75).item())
            e_emb = self.energy_embedding(e_emb) # # (B, T, d/2)
            e_emb = F.interpolate(e_emb.transpose(-1, -2), size=seq_style.size(-1), mode="linear", align_corners=True)
            #e_emb = self.conv_layer(e_emb)  # (B, d/2, T)

        # Do log first
        #pitch_safe = torch.clamp(seq_style_gt[:, 1, :], min=eps)
        #pitch_safe = torch.log(pitch_safe)
        p_emb = torch.bucketize(seq_style_gt[:, 1, :], self.pitch_bins)  # (B, T)
        if self.VIS_PITCH_REAL_DISCRETE and not self.training:
            from exp.vis2 import plot_f0_comparison
            show_num = min(5, seq_style_gt.size(0))
            for i in range(show_num):
                plot_f0_comparison(seq_style_gt[i, 1], p_emb[i], seq_style[i, 1],
                                   out_path=f"res/temp/pitch_real_discrete_pred_{i}.png",
                                   labels=("pitch_real", "pitch_discrete", "pitch_pred"))
                plot_f0_comparison(seq_style_gt[i, 0], p_emb[i], seq_style[i, 0],
                                   out_path=f"res/temp/energy_real_discrete_pred_{i}.png",
                                   labels=("energy_real", "energy_discrete", "energy_pred"))
            self.VIS_PITCH_REAL_DISCRETE = False
        #print("p_emb q25, q75", torch.quantile(p_emb[0].float(), q=0.25).item(), torch.quantile(p_emb[0].float(), q=0.75).item())
        p_emb = self.pitch_embedding(p_emb)
        p_emb = F.interpolate(p_emb.transpose(-1, -2), size=seq_style.size(-1), mode="linear", align_corners=True)
        #p_emb = self.conv_layer(p_emb)  # (B, d/2, T)

        # encode
        seq_style = self.pe_encode(seq_style)  # (B, d, T)  d =256
        #seq_style_gt = self.pe_phone_encode(torch.cat((e_emb, p_emb), dim=1))  # (B, d, T)
        seq_style_gt = self.pe_phone_encode(p_emb)  # (B, d, T)

        # ----- gate (no-op at init) -----
        gate = torch.sigmoid(self.gd_gate)  # scalar in (0,1)
        seq_style_gt = gate * seq_style_gt

        # fusion
        seq_style = self.proj(torch.cat((seq_style, seq_style_gt), dim=1).transpose(-1, -2)).transpose(-1, -2)  # (B, d, T)
        return seq_style
