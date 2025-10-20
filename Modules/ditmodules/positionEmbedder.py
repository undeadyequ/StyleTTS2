import torch
import torch.nn as nn
import torch.nn.functional as F

class PhoneRotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module
    https://nn.labml.ai/transformers/rope/index.html

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor, seq_idx: torch.Tensor):
        """
        x: [seq_len, batch_size, n_heads, d]
        seq_dur: [batch_size, 1, seq_len]

        Cache $\cos$ and $\sin$ values

        """
        # Return if cache is already built
        #if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
        #    return

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)  # bd
        theta = theta.repeat(x.shape[1], 1)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        if seq_idx is not None:
            seq_idx1 = seq_idx.float().to(x.device)
            # Calculate the product of position index and $\theta_i$
        else:
            # Get sequence length
            seq_len = x.shape[0]
            seq_idx1 = torch.arange(seq_len, device=x.device).float().to(x.device)   # (seq_len, )
            seq_idx1 = seq_idx1.repeat(x.shape[1], 1)  # (batch_size, seq_len)
            ######################## MANIPULATE ################
            #seq_idx1[:min(40, len(seq_idx1))] += 80
            ## print(seq_idx)
            ####################################################
            # Calculate the product of position index and $\theta_i$

        idx_theta = torch.einsum("bn,bd->bnd", seq_idx1, theta)     # (b, seq_len, dim/2)
        idx_theta = idx_theta.permute(1, 0, 2)                            # (seq_len, b, dim/2)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=2)   # (seq_len, b, dim)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, :, None, :]            # (seq_len, b, 1, dim)
        self.sin_cached = idx_theta2.sin()[:, :, None, :]            # (seq_len, b, 1, dim)

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor, seq_dur=None):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        seq_dur = [batch_size, seq_len]
        """
        # Cache $\cos$ and $\sin$ values
        x = x.permute(2, 0, 1, 3)  # b h t d -> t b h d

        self._build_cache(x, seq_dur)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d:]  # (seq, b,  h,  d)

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)  # negative of half x_rope

        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return torch.cat((x_rope, x_pass), dim=-1).permute(1, 2, 0, 3)  # t b h d -> b h t d


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        """
        def _buid_cache(self, x, p_dur):
            p_dur: [p_len, batch_size]
        ...
        if phone_rope:
            p_dur_seqLen = convert_pDur_seqLen(p_dur)  #  [p_len, batch_size] -> [f_len, batch_size]
            seq_idx = torch.arange(p_dur, device).float().to(x.device)
        else:
            seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        """
        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        x = x.permute(2, 0, 1, 3)  # b h t d -> t b h d

        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d:]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return torch.cat((x_rope, x_pass), dim=-1).permute(1, 2, 0, 3)  # t b h d -> b h t d