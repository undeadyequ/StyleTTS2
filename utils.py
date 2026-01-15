from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch
import os
from typing import Tuple


def maximum_path(neg_cent, mask):
    """ Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
    t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)


# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def append_sentence_to_file(file_path, sentence):
    """
    Append a sentence to the end of a file.
    If the file does not exist, it will be created automatically.

    Args:
        file_path (str): Path to the file.
        sentence (str): The sentence to append.
    """
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    # Open the file in append mode ("a" creates it if it doesn't exist)
    with open(file_path, "a", encoding="utf-8") as f:
        # Add newline if file already has content
        if os.path.getsize(file_path) > 0:
            f.write("\n")
        f.write(sentence)
    print(f"Sentence appended successfully to {file_path!r}.")


def frame_to_phoneme_avg_and_back_binary(
    energy_f: torch.Tensor,  # [B, Tf]
    pitch_f: torch.Tensor,   # [B, Tf]
    attn_f2p: torch.Tensor,  # [B, Tp, Tf], entries in {0,1}
    eps: float = 1e-8,
):
    attn_f2p = attn_f2p.transpose(1, 2)
    # w is binary assignment mask
    w = attn_f2p.to(pitch_f.dtype)

    # frame -> phoneme: mean over assigned frames
    denom_p = w.sum(dim=1)  # [B, Tp] = durations
    #print(denom_p[0])
    pitch_p = torch.einsum("bt,btp->bp", pitch_f, w) / (denom_p + eps)
    energy_p = torch.einsum("bt,btp->bp", energy_f, w) / (denom_p + eps)
    #print("pitch_phoneme", pitch_p[0])
    #print("w", w[0])

    # phoneme -> frame: repeat assigned phoneme value
    # (since each frame belongs to exactly one phoneme, this is exact)

    pitch_f_back = torch.einsum("bp,btp->bt", pitch_p, w)
    energy_f_back = torch.einsum("bp,btp->bt", energy_p, w)
    #print(pitch_f_back[0])

    return energy_p, pitch_p, energy_f_back, pitch_f_back

def make_binary_attn_from_durations(durations: torch.Tensor) -> torch.Tensor:
    """
    durations: [B, Tp] ints, sum over Tp = Tf
    returns attn: [B, Tf, Tp] with 0/1, each frame assigned to exactly one phoneme
    """
    B, Tp = durations.shape
    Tf = int(durations.sum(dim=1).max().item())
    assert (durations.sum(dim=1) == Tf).all(), "All batch items must have same Tf in this simple test."

    attn = torch.zeros(B, Tf, Tp, dtype=torch.int64)
    for b in range(B):
        t = 0
        for p in range(Tp):
            d = int(durations[b, p].item())
            attn[b, t:t+d, p] = 1
            t += d
    return attn


def align_dur(pe_real, pe_uv_mask):
    """
    1. interpolate unvoiced frames on pe_real
    2. apply pe_uv_mask to the interpolated pe_real
    pe_real: (B, 2, T)
    pe_uv_mask: (B, 1, T)
    align duration mismatch between pe_real and pe given pe_uv_mask
    """
    voiced_pe_real = (pe_real[:, 1, :] > 1e-3).unsqueeze(1).float()
    voiced_pe_real_interp = fill_unvoiced_with_interp(pe_real, voiced_pe_real)
    voiced_pe_real_interp = F.interpolate(voiced_pe_real_interp[None, None, :], size=pe_uv_mask.size(), mode="linear",
                                          align_corners=True).squeeze()

    voiced_pe_real_interp = voiced_pe_real_interp * pe_uv_mask
    return voiced_pe_real_interp


def fill_unvoiced_with_interp(f0, voiced):
    idx = torch.arange(len(f0), device=f0.device)
    voiced_idx = idx[voiced.bool()]
    voiced_vals = f0[voiced.bool()]
    if len(voiced_idx) < 2:
        return f0.clone()
    f0_interp_np = torch.from_numpy(
        __import__('numpy').interp(
            idx.cpu().numpy(),
            voiced_idx.cpu().numpy(),
            voiced_vals.cpu().numpy()))
    return f0_interp_np.to(f0.device, dtype=f0.dtype)


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


def align_dur2(
    pe_real: torch.Tensor,     # [B, 2, T_real] (energy, pitch) or (pitch, energy) -> set pitch_idx accordingly
    pe_uv_mask: torch.Tensor,  # [B, 1, T_mask] predicted voiced mask (1=voiced, 0=unvoiced)
    pitch_idx: int = 1,        # index of pitch channel in pe_real
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    1) Compute voiced mask from GT pitch channel pe_real[:, pitch_idx, :].
    2) Interpolate unvoiced frames of BOTH channels using this voiced mask.
    3) Apply predicted voiced mask pe_uv_mask to the interpolated result (multiply).
    Returns: [B, 2, T]
    """
    assert pe_real.ndim == 3 and pe_real.size(1) == 2
    assert pe_uv_mask.ndim == 3 and pe_uv_mask.size(1) == 1
    assert pe_real.size(0) == pe_uv_mask.size(0)

    B, _, T_real = pe_real.shape
    T_mask = pe_uv_mask.size(-1)

    # ------------------------------------------------
    # Step 1: resample pe_real if needed
    # ------------------------------------------------
    if T_real != T_mask:
        # F.interpolate expects [B, C, T]
        pe_real = F.interpolate(
            pe_real,
            size=T_mask,
            mode="linear",
            align_corners=True,
        )

    # ------------------------------------------------
    # Step 2: voiced mask from GT pitch (LibriTTS-safe)
    # ------------------------------------------------
    pitch_gt = pe_real[:, pitch_idx, :]          # [B, T]
    voiced_gt = (pitch_gt >= 50.0) & (pitch_gt <= 600.0) & torch.isfinite(pitch_gt)  # [B, T] bool (assumes unvoiced pitch == 0)

    # ------------------------------------------------
    # Step 3: interpolate unvoiced regions
    # ------------------------------------------------

    out = pe_real.clone()

    # ------------------------------------------------
    # Step 4: apply predicted voiced mask
    # ------------------------------------------------

    out[:, 0, :] = _interp_unvoiced_1d(out[:, 0, :], voiced_gt)  # energy filled over unvoiced regions
    out[:, 1, :] = _interp_unvoiced_1d(out[:, 1, :], voiced_gt)  # pitch filled over unvoiced regions

    # Apply predicted voiced mask (broadcast over channel)
    out = out * pe_uv_mask.to(out.dtype)  # [B,2,T]
    return out


def extract_pitch_trend(
    pitch_f: torch.Tensor,   # [B, Tf]  (Hz)
    attn_f2p: torch.Tensor,  # [B, Tp, Tf] 0/1
    tareget_len: int,
    eps: float = 1e-8,
    fmin: float = 50.0,
    fmax: float = 600.0,
):
    """
    Extract phoneme-level pitch trend in log-F0 space.

    Steps:
      0) voiced mask from raw F0
      1) log-F0 transform
      2) frame -> phoneme mean (log domain)
      3) phoneme -> frame broadcast
      4) interpolate unvoiced gaps
      5) resample to tareget_len

    Return:
      log_f0_trend: [B, 1, tareget_len]
    """
    attn_f2p = F.interpolate(attn_f2p, size=pitch_f.size(-1), mode="nearest")

    assert pitch_f.ndim == 2, f"pitch_f must be [B,Tf], got {pitch_f.shape}"
    assert attn_f2p.ndim == 3, f"attn_f2p must be [B,Tp,Tf], got {attn_f2p.shape}"
    assert pitch_f.size(0) == attn_f2p.size(0)
    assert pitch_f.size(1) == attn_f2p.size(2)

    B, Tf = pitch_f.shape

    # ------------------------------------------------
    # Step 0: voiced mask from raw F0 (robust to JDC noise)
    # ------------------------------------------------
    voiced = (
        (pitch_f >= fmin) &
        (pitch_f <= fmax) &
        torch.isfinite(pitch_f)
    )  # [B, Tf]

    # ------------------------------------------------
    # Step 1: log-F0
    # ------------------------------------------------
    pitch_f_safe = pitch_f.clamp_min(fmin)
    log_f0 = torch.log(pitch_f_safe)  # [B, Tf]

    # ------------------------------------------------
    # Step 2: frame -> phoneme mean (log domain)
    # ------------------------------------------------
    w = attn_f2p.transpose(1, 2).to(log_f0.dtype)   # [B, Tf, Tp]
    denom = w.sum(dim=1).clamp_min(1.0)             # [B, Tp]

    log_f0_p = torch.einsum("bt,btp->bp", log_f0, w) / (denom + eps)  # [B, Tp]

    # ------------------------------------------------
    # Step 3: phoneme -> frame broadcast
    # ------------------------------------------------
    log_f0_back = torch.einsum("bp,btp->bt", log_f0_p, w)  # [B, Tf]
    log_f0_back = log_f0_back.unsqueeze(1)                # [B,1,Tf]

    # ------------------------------------------------
    # Step 4: interpolate unvoiced gaps
    # ------------------------------------------------
    log_f0_filled = log_f0_back.clone()
    log_f0_filled[:, 0, :] = _interp_unvoiced_1d(
        log_f0_back[:, 0, :], voiced
    )

    # ------------------------------------------------
    # Step 5: resample to bert length
    # ------------------------------------------------
    if log_f0_filled.size(-1) != tareget_len:
        log_f0_filled = F.interpolate(
            log_f0_filled,
            size=tareget_len,
            mode="linear",
            align_corners=True,
        )

    return log_f0_filled


def make_attn_f2p_from_durations(durs: torch.Tensor) -> torch.Tensor:
    """
    durs: [B, Tp] integer durations summing to Tf
    returns attn_f2p: [B, Tp, Tf] with 0/1
    """
    B, Tp = durs.shape
    Tf = int(durs.sum(dim=1)[0].item())
    attn = torch.zeros(B, Tp, Tf, device=durs.device, dtype=torch.float32)
    for b in range(B):
        t0 = 0
        for p in range(Tp):
            d = int(durs[b, p].item())
            attn[b, p, t0:t0+d] = 1.0
            t0 += d
        assert t0 == Tf
    return attn


def test_extract_pitch_trend_logf0(device="cpu"):
    torch.manual_seed(0)
    B, Tp = 2, 6

    # durations -> Tf (keep Tf reasonably large for visualization; but still robust)
    durs = torch.randint(6, 12, (B, Tp), device=device)
    durs[1] = durs[0]
    Tf = int(durs[0].sum().item())

    attn_f2p = make_attn_f2p_from_durations(durs)  # [B,Tp,Tf]

    # synthetic pitch (Hz)
    t = torch.linspace(0, 1, Tf, device=device)
    pitch = 180.0 + 40.0 * torch.sin(2 * torch.pi * 2 * t)  # [Tf]
    pitch = pitch.unsqueeze(0).repeat(B, 1)                 # [B,Tf]

    # Insert "unvoiced junk" segments safely within Tf
    def set_junk(start: int, length: int, max_hz: float):
        start = max(0, min(start, Tf))
        end = max(start, min(start + length, Tf))
        if end > start:
            pitch[:, start:end] = torch.rand(B, end - start, device=device) * max_hz

    # two junk regions (sizes auto-adjust if Tf is small)
    set_junk(start=Tf // 4, length=max(2, Tf // 10), max_hz=3.0)
    set_junk(start=Tf * 3 // 4, length=max(2, Tf // 12), max_hz=15.0)

    tareget_len = Tp
    log_trend = extract_pitch_trend(pitch, attn_f2p, tareget_len)

    print("Tf =", Tf)
    print("pitch:", pitch.shape)
    print("attn_f2p:", attn_f2p.shape)
    print("log_trend:", log_trend.shape)

    assert log_trend.shape == (B, 1, tareget_len)
    assert torch.isfinite(log_trend).all()
    print("OK")


if __name__ == '__main__':
    B, Tp = 2, 6
    Tf = 5
    """
    attn = torch.tensor([[
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]], dtype=torch.int64)

    # sanity: each frame assigned to exactly 1 phoneme

    pitch_f = 50 + 250 * torch.rand(B, Tf)
    energy_f = torch.rand(B, Tf)

    pitch_p, energy_p, pitch_f_back, energy_f_back = frame_to_phoneme_avg_and_back_binary(
        pitch_f, energy_f, attn
    )

    print("pitch", pitch_f[0])
    print("pitch_phoneme", pitch_p[0])
    print("pitch back", pitch_f_back[0])

    print("energy", energy_f[0])
    print("energy phoneme", energy_p[0])
    print("energy back", energy_f_back)
    """
    """
    from exp.vis2 import plot_f0_comparison
    B, T = 2, 12
    pe_real = torch.randn(B, 2, T).abs()

    pe_real[:, 1, :] *= 200.0  # pretend channel 1 is pitch
    # make some unvoiced in gt pitch
    pe_real[:, 1, 3:6] = 0.5

    pe_uv_mask = torch.ones(B, 1, T)
    pe_uv_mask[:, :, 8:] = 0.0   # predicted unvoiced tail

    pe_proc = align_dur2(pe_real, pe_uv_mask)

    plot_f0_comparison(pe_real[0, 1], pe_uv_mask[0, 0] * 20, pe_proc[0, 1],
                       out_path="res/hierstyle_cond/pitch_compare.png", labels=("real", "pred_mask", "dur_aligned_real"))

    print(pe_real)  # [B,2,T]
    print(pe_proc)
    """
    test_extract_pitch_trend_logf0(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
