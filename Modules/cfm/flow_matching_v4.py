"""CFM decoder with masked-mel conditioning (v4).

Key differences from flow_matching_v3.py (CFMDecoderV3):

  1. Global speaker embedding c removed.
     Speaker/style identity is supplied via `cond` (masked reference mel).

  2. Random span masking during training (F5-TTS style):
     - A contiguous span of frames is randomly chosen as the "target" region.
     - That region is zeroed out to produce `cond`.
     - MSE loss is computed only on the masked (target) frames.

  3. CFG via cfg_wrapper (same pattern as v3):
     - During training: with probability cfg_dropout, ALL conditioning signals
       (cond, mu, seq_style) are replaced by learned null embeddings, so the
       model also learns fully-unconditional generation.
     - During inference: cfg_wrapper does a dual forward pass and returns
       v_uncond + cfg_strength * (v_cond - v_uncond).

  4. Inference uses F5-TTS ref+tgt temporal concatenation:
     - ODE initial state z = concat([ref_mel, noise_tgt], dim=-1)
       (ref stays clean at t=1; tgt starts from noise at t=0).
     - cond_full = concat([ref_mel, zeros_tgt], dim=-1)  (tgt region zeroed).
     - mu_full   = concat([mu_ref,  mu_tgt],    dim=-1)
     - seq_style_full = concat([seq_style_ref, seq_style_tgt], dim=-1)
     - After ODE: mel = trajectory[-1][:, :, T_ref:]  (extract tgt portion only).

Select via cfm_config in the YAML:
    cfm_config:
        use_v4: true
        cross_attn: true
        cfg_dropout: 0.2
        ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from torchdiffeq import odeint
from torch.nn.utils import weight_norm

from Modules.ditmodules.estimator_v4 import DecoderV4
from Modules.latent_diffusion.pre_dit_modules import ResBlk1d


class CFMDecoderV4(torch.nn.Module):
    """Flow-matching decoder with masked-mel conditioning (v4).

    Args:
        noise_channels   : mel spectrogram channels (e.g. 80 after asr_res)
        cond_channels    : mu channels before asr_res (e.g. 512)
        hidden_channels  : DiT hidden dim
        out_channels     : output mel channels (= noise_channels)
        filter_channels  : FFN / conv filter dim
        n_heads          : attention heads
        n_layers         : number of DiT blocks (must be even when use_lsc=True)
        kernel_size      : conv kernel size
        p_dropout        : dropout probability
        cross_attn       : whether to use cross-attention for seq_style
        residual_dim     : ResBlk1d intermediate dim for pe_encode
        dim_in           : input dim of mu before asr_res projection
        use_lsc          : long-skip connections
        mask_ratio_min   : minimum fraction of frames to mask during training
        mask_ratio_max   : maximum fraction of frames to mask during training
        cfg_dropout      : probability of replacing ALL conditioning signals with
                           learned null embeddings during training (F5-TTS style CFG)
    """

    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 cross_attn, residual_dim=64, dim_in=512,
                 use_lsc=True, mask_ratio_min=0.7, mask_ratio_max=1.0,
                 cfg_dropout=0.0):
        super().__init__()

        self.noise_channels  = noise_channels
        self.cond_channels   = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.sigma_min       = 1e-4
        self.cross_attn      = cross_attn
        self.mask_ratio_min  = mask_ratio_min
        self.mask_ratio_max  = mask_ratio_max
        self.cfg_dropout     = cfg_dropout

        self.estimator = DecoderV4(
            noise_channels, cond_channels, hidden_channels, out_channels,
            filter_channels, p_dropout, n_layers, n_heads, kernel_size,
            cross_attn=cross_attn, use_lsc=use_lsc)

        pe_emb_dim  = 256
        asr_res_dim = noise_channels   # asr_res output matches mel channels

        self.pe_encode = nn.Sequential(
            ResBlk1d(2,           residual_dim, normalize=True),
            ResBlk1d(residual_dim, pe_emb_dim,  normalize=True))

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, asr_res_dim, kernel_size=3, padding=1)),
            nn.InstanceNorm1d(asr_res_dim, affine=True))

        # Learned null embeddings for CFG (initialised to zero, broadcastable)
        if cfg_dropout > 0:
            self.fake_cond = nn.Parameter(torch.zeros(1, noise_channels, 1))
            self.fake_mu   = nn.Parameter(torch.zeros(1, asr_res_dim,    1))
            self.fake_pe   = nn.Parameter(torch.zeros(1, pe_emb_dim,     1))

        self.attn_cache = []
        self.t_count    = []

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _make_span_mask(self, x1):
        """Random contiguous span masking.

        Args:
            x1: [B, C, T]  clean mel spectrogram
        Returns:
            cond      : [B, C, T]  x1 with target region zeroed
            loss_mask : [B, 1, T]  1 in masked (target) region, 0 elsewhere
        """
        B, C, T = x1.shape
        device  = x1.device

        # Sample mask length as a fraction of T
        frac    = torch.empty(B, device=device).uniform_(
            self.mask_ratio_min, self.mask_ratio_max)
        lengths = (frac * T).long().clamp(min=1, max=T - 1)

        # Sample start positions (ensure start + length <= T)
        max_starts = (T - lengths).clamp(min=0)                          # [B]
        starts     = (torch.rand(B, device=device)
                      * (max_starts.float() + 1)).long().clamp(max=max_starts)

        # Build boolean mask: [B, T]
        idx       = torch.arange(T, device=device).unsqueeze(0)          # [1, T]
        starts_2d = starts.unsqueeze(1)                                   # [B, 1]
        ends_2d   = (starts + lengths).unsqueeze(1)                      # [B, 1]
        loss_mask = ((idx >= starts_2d) & (idx < ends_2d)).float().unsqueeze(1)  # [B,1,T]

        cond = x1 * (1.0 - loss_mask)   # zero out target region
        return cond, loss_mask

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, x1, mu, seq_style=None):
        """Compute flow-matching loss with random span masking.

        Args:
            x1        : [B, C, T]        clean mel spectrogram
            mu        : [B, dim_in, T/2] frame-level text embedding (pre-interpolate)
            seq_style : [B, 2, T]        pitch/energy contours
        Returns:
            loss      : scalar
            None      : placeholder for attn_maps (matches v3 API)
        """
        mu        = F.interpolate(mu, scale_factor=2, mode="nearest")
        mu        = self.asr_res(mu)          # [B, noise_channels, T]
        seq_style = self.pe_encode(seq_style) # [B, 256, T]

        # Random span mask
        cond, loss_mask = self._make_span_mask(x1)   # [B,C,T], [B,1,T]

        b, _, t = mu.shape

        # CFG dropout: replace ALL conditioning signals with learned null embeddings
        # so the model learns fully-unconditional generation (F5-TTS cond_drop_prob).
        if self.cfg_dropout > 0.0:
            cfg_mask  = (torch.rand(b, 1, 1, device=cond.device) > self.cfg_dropout)
            cond      = cond      * cfg_mask + (~cfg_mask) * self.fake_cond.expand(b, -1, cond.size(-1))
            mu        = mu        * cfg_mask + (~cfg_mask) * self.fake_mu.expand(b,   -1, mu.size(-1))
            seq_style = seq_style * cfg_mask + (~cfg_mask) * self.fake_pe.expand(b,   -1, seq_style.size(-1))

        # Cosine timestep schedule
        time = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        time = 1 - torch.cos(time * 0.5 * torch.pi)

        # Flow interpolation
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * time) * z + time * x1   # x_t
        u = x1 - (1 - self.sigma_min) * z                        # target velocity

        x_mask   = torch.ones([b, 1, t], device=x1.device)
        p_mask_i = torch.ones([b, 1, seq_style.size(-1)], device=x1.device)

        estm_out = self.estimator(time.squeeze(), y, cond, x_mask, mu, seq_style, p_mask_i)

        # Loss only on masked (target) region
        n    = (loss_mask.sum() * u.size(1)).clamp(min=1)
        loss = F.mse_loss(estm_out * loss_mask, u * loss_mask, reduction="sum") / n

        return loss, None

    # ------------------------------------------------------------------
    # CFG wrapper
    # ------------------------------------------------------------------

    def cfg_wrapper(self, t, x, cond, mask, mu, cfg_strength,
                    seq_style=None, p_mask=None):
        """Dual-pass estimator for classifier-free guidance.

        Conditioned pass uses the real cond / mu / seq_style.
        Unconditioned pass replaces all three with learned null embeddings.
        Returns: v_uncond + cfg_strength * (v_cond - v_uncond)
        """
        B = x.size(0)
        fake_cond = self.fake_cond.expand(B, -1, x.size(-1))
        fake_mu   = self.fake_mu.expand(B,   -1, mu.size(-1))
        fake_pe   = self.fake_pe.expand(B,   -1, seq_style.size(-1))

        v_cond   = self.estimator(t, x, cond,      mask, mu,      seq_style, p_mask)
        v_uncond = self.estimator(t, x, fake_cond, mask, fake_mu, fake_pe,   p_mask)

        return v_uncond + cfg_strength * (v_cond - v_uncond)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, mu_tgt, n_timesteps, temperature=1.0,
                cond_ref=None, mu_ref=None,
                seq_style_tgt=None, seq_style_ref=None,
                solver=None, cfg_strength=None, return_attn_map=True):
        """ODE-based inference with F5-TTS ref+tgt temporal concatenation.

        Args:
            mu_tgt        : [B, dim_in, T_tgt/2]  target frame-level text embedding
            n_timesteps   : ODE steps
            temperature   : noise scale for initial target noise
            cond_ref      : [B, C, T_ref]          reference mel frames (clean)
            mu_ref        : [B, dim_in, T_ref/2]   reference frame-level text embedding
            seq_style_tgt : [B, 2, T_tgt]          target pitch/energy contours
            seq_style_ref : [B, 2, T_ref]          reference pitch/energy contours
            solver        : torchdiffeq solver name (default: dopri5)
            cfg_strength  : CFG scale; None (default) = no CFG. Pass e.g. 2.0 to
                            amplify reference conditioning (requires cfg_dropout > 0
                            during training).
        Returns:
            mel  : [B, C, T_tgt]  synthesised mel
            attn : attention maps or None
        """
        # Project target text embedding
        mu_tgt = F.interpolate(mu_tgt, scale_factor=2, mode="nearest")
        mu_tgt = self.asr_res(mu_tgt)            # [B, noise_channels, T_tgt]

        # Project reference text embedding
        mu_ref = F.interpolate(mu_ref, scale_factor=2, mode="nearest")
        mu_ref = self.asr_res(mu_ref)            # [B, noise_channels, T_ref]

        # Project pitch/energy contours
        seq_style_tgt = self.pe_encode(seq_style_tgt)   # [B, 256, T_tgt]
        seq_style_ref = self.pe_encode(seq_style_ref)   # [B, 256, T_ref]

        B      = mu_tgt.size(0)
        T_tgt  = mu_tgt.size(-1)
        T_ref  = cond_ref.size(-1)
        T_full = T_ref + T_tgt
        device = mu_tgt.device

        # ODE initial state: ref stays clean (t=1), tgt starts from noise (t=0)
        noise_tgt = torch.randn(B, self.noise_channels, T_tgt, device=device) * temperature
        z = torch.cat([cond_ref, noise_tgt], dim=-1)   # [B, C, T_full]

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)

        # Concatenated mel_ref (clean) and mel_tgt (zeros, fixed throughout ODE)
        cond_full = torch.cat(
            [cond_ref, torch.zeros(B, self.noise_channels, T_tgt, device=device)],
            dim=-1)   # [B, C, T_full]

        # Concatenated mu, seq_style along time dim
        mu_full        = torch.cat([mu_ref,        mu_tgt],        dim=-1)  # [B, C, T_full]
        seq_style_full = torch.cat([seq_style_ref, seq_style_tgt], dim=-1)  # [B, 256, T_full]

        x_mask      = torch.ones([B, 1, T_full],                  device=device)
        p_mask_full = torch.ones([B, 1, seq_style_full.size(-1)], device=device)

        # odeint requires f(t, x) -> tensor
        if cfg_strength is not None:
            if self.cfg_dropout <= 0:
                raise IOError("cfg_dropout must be > 0 during training for CFG inference")
            estimator_fn = functools.partial(
                self.cfg_wrapper,
                cond=cond_full, mask=x_mask, mu=mu_full, cfg_strength=cfg_strength,
                seq_style=seq_style_full, p_mask=p_mask_full)
        else:
            estimator_fn = functools.partial(
                self.estimator,
                cond=cond_full, mask=x_mask, mu=mu_full,
                seq_style=seq_style_full, p_mask=p_mask_full,
                return_attn_map=False)
        trajectory = odeint(estimator_fn, z, t_span,
                            method=solver, rtol=1e-5, atol=1e-5)
        # Extract only the target portion from the final trajectory state
        mel = trajectory[-1][:, :, T_ref:]   # [B, C, T_tgt]

        # Optional: single forward pass to collect attention maps
        attn_maps = None
        if return_attn_map:
            _, attn_maps = self.estimator(
                t_span[-2], trajectory[-1], cond_full, x_mask, mu_full,
                seq_style_full, p_mask_full,
                return_attn_map=True)

        return mel, attn_maps
