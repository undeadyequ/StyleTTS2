import torch
import torch.nn as nn
from drawspeech.modules.ditmodules.quantize.rvq import ResidualVQ

class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x

# modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/module/modules.py#L766
class MelStyleEncoderSeq(nn.Module):
    """MelStyleEncoderSeq: without average than MelStyleEncoder"""

    def __init__(
        self,
        n_mel_channels=80,
        style_hidden=128,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
        num_quantizers=1,   # quantizer
        vq_dim=256,
        codebook_size=10,
        codebook_dim=8,
        vq_threshold_ema_dead_code=2,
        vq_commit_weight=0.003,
        vq_weight_init=None,
        vq_full_commit_loss=True,
        vq_quantizer_dropout=0.0,
        vq_dropout_type="linear",
    ):
        super(MelStyleEncoderSeq, self).__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        # spectral embedding
        self.spectral = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
        )
        # temporal embedding
        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )
        # quantizer
        quanizer_class = ResidualVQ
        self.f0_quantizer = quanizer_class(
            num_quantizers=num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=vq_threshold_ema_dead_code,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
            quantizer_dropout=vq_quantizer_dropout,
            dropout_type=vq_dropout_type,
        )
        self.fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, x_mask=None, n_quantizers=None):
        """
        x: (batch_size, n_feats, max_mel_length)
        return:
            shape: ...
        """
        x = x.transpose(1, 2)
        # Spectral
        x = self.spectral(x)
        # Temporal
        x = x.transpose(1, 2)
        x = self.temporal(x * x_mask)
        x = x * x_mask

        # Quantizer
        out, q, commit, quantized = self.f0_quantizer(x, n_quantizers=n_quantizers)
        # fc
        quantized = quantized.sum(0)  # add residual quants
        x = self.fc(quantized.transpose(1, 2)).transpose(1, 2) * x_mask
        return x, commit


# modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/module/modules.py#L766    
class MelStyleEncoder(nn.Module):
    """MelStyleEncoder"""

    def __init__(
        self,
        n_mel_channels=80,
        style_hidden=128,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
    ):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = nn.MultiheadAttention(
            self.hidden_dim,
            self.n_head,
            self.dropout,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            return torch.sum(x * ~mask.unsqueeze(-1), dim=1) / (~mask).sum(dim=1).unsqueeze(1)

    def forward(self, x, x_mask=None):
        x = x.transpose(1, 2)
        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        if x_mask is not None:
            x_mask = ~x_mask.squeeze(1).to(torch.bool)   
        x, _ = self.slf_attn(x, x, x, key_padding_mask=x_mask, need_weights=False)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=x_mask)

        return w
    
# Attention Pool version of MelStyleEncoder, not used
class AttnMelStyleEncoder(nn.Module):
    """MelStyleEncoder"""

    def __init__(
        self,
        n_mel_channels=80,
        style_hidden=128,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
    ):
        super().__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = nn.MultiheadAttention(
            self.hidden_dim,
            self.n_head,
            self.dropout,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.out_dim)
        
    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            return torch.sum(x * ~mask.unsqueeze(-1), dim=1) / (~mask).sum(dim=1).unsqueeze(1)

    def forward(self, x, x_mask=None):
        x = x.transpose(1, 2)

        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        if x_mask is not None:
            x_mask = ~x_mask.squeeze(1).to(torch.bool)
            zeros = torch.zeros(x_mask.size(0), 1, device=x_mask.device, dtype=x_mask.dtype)
            x_attn_mask = torch.cat((zeros, x_mask), dim=1)
        else:
            x_attn_mask = None

        avg = self.temporal_avg_pool(x, x_mask).unsqueeze(1)
        x = torch.cat([avg, x], dim=1)
        x, _ = self.slf_attn(x, x, x, key_padding_mask=x_attn_mask, need_weights=False)
        x = x[:, 0, :]
        # fc
        x = self.fc(x)

        return x