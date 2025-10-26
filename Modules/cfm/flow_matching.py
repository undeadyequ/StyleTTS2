import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchdiffeq import odeint

from Modules.ditmodules.estimator import Decoder
from Modules.latent_diffusion.pre_dit_modules import ResBlk1d
from torch.nn.utils import weight_norm


# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py



class CFMDecoder(torch.nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout, gin_channels, cross_attn, residual_dim=64, dim_in=512):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 1e-4
        self.cross_attn = cross_attn
        self.estimator = Decoder(noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, p_dropout, n_layers,
                                 n_heads, kernel_size, gin_channels, cross_attn=cross_attn)

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


    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None, seq_style=None, p_mask=None, solver=None, cfg_kwargs=None,
                q_f_pos=None, k_f_pos=None):
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
            cfg_kwargs: used for cfg inference

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        mu = F.interpolate(mu, scale_factor=2, mode="nearest")
        mu = self.asr_res(mu)
        seq_style = self.pe_encode(seq_style)

        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        mask = torch.ones([z.size(0), 1, z.size(-1)]).to(z.device)
        p_mask = mask

        # cfg control
        if cfg_kwargs is None:
            ### TODO-S: tempt code for returning attn_map of dit at t=0
            _, attn_maps = self.estimator(t_span[0], z, mask=mask, mu=mu, c=c,seq_style=seq_style, p_mask=p_mask, return_attn_map=True)
            estimator = functools.partial(self.estimator, mask=mask, mu=mu, c=c,seq_style=seq_style, p_mask=p_mask, return_attn_map=False)
        else:
            _, attn_maps = self.cfg_wrapper(t_span[0], z, mask=mask, mu=mu, c=c, cfg_kwargs=cfg_kwargs, seq_style=seq_style, p_mask=p_mask, return_attn_map=True)
            estimator = functools.partial(self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_kwargs=cfg_kwargs, seq_style=seq_style, p_mask=p_mask, return_attn_map=False)
        ### TODO-B
        trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)
        return trajectory[-1], attn_maps
    
    # cfg inference
    def cfg_wrapper(self, t, x, mask, mu, c, cfg_kwargs, seq_style=None, p_mask=None, return_attn_map=False):
        fake_speaker = cfg_kwargs['fake_speaker'].repeat(x.size(0), 1)
        fake_content = cfg_kwargs['fake_content'].repeat(x.size(0), 1, x.size(-1))
        fake_psd = cfg_kwargs['fake_psd'].repeat(seq_style.size(0), 1, seq_style.size(-1))

        cfg_strength = cfg_kwargs['cfg_strength']
        
        cond_output = self.estimator(t, x, mask, mu, c, seq_style, p_mask, return_attn_map)
        uncond_output = self.estimator(t, x, mask, fake_content, fake_speaker, fake_psd, p_mask, return_attn_map)
        #uncond_output = self.estimator(t, x, mask, fake_content, fake_speaker, None, None, return_attn_map)

        output = uncond_output + cfg_strength * (cond_output - uncond_output)
        return output

    def compute_loss(self, x1, mask, mu, c, seq_style=None, p_mask=None):
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
        seq_style = self.pe_encode(seq_style)

        b, _, t = mu.shape

        # random timestep
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

        estm_out, attn_maps = self.estimator(t.squeeze(), y, x_mask, mu, c, seq_style, p_mask)
        loss = F.mse_loss(estm_out, u, reduction="sum") / (torch.sum(x_mask) * u.size(1))
        #return loss, y
        #return loss, estm_out, attn_maps
        return loss, attn_maps         ######### TEMP
