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
from torch.nn.utils.rnn import pad_sequence

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
    Extract frame-level pitch trend in log-F0 space.

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
    attn_f2p = F.interpolate(attn_f2p, size=pitch_f.size(-1), mode="nearest")  # critical problem: check if pitch_f is twice length of attn_f2p first, and then upsample

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


def extract_pitch_trend_v2(
        pitch_gd_frame: torch.Tensor,  # [B, Tf]  (Hz)
        attn_f2p_gd: torch.Tensor,  # [B, Tp, Tf] 0/1
        tgt_frame_len: int,
        attn_f2p_pred: torch.Tensor = None,  # [B, Tp_syn, Tf_syn] 0/1
        eps: float = 1e-8,
        fmin: float = 50.0,
        fmax: float = 600.0,
):
    """
    Extract phoneme-level pitch trend in log-F0 space.
    """
    # Ensure ground truth attention matches the frame length of reference audio
    attn_f2p_gd = F.interpolate(attn_f2p_gd, size=pitch_gd_frame.size(-1), mode="nearest")
    B, Tf = pitch_gd_frame.shape

    # ------------------------------------------------
    # Step 0 & 1: Voice Masking and Log Transform
    # ------------------------------------------------
    voiced = (pitch_gd_frame >= fmin) & (pitch_gd_frame <= fmax) & torch.isfinite(pitch_gd_frame)
    pitch_f_safe = pitch_gd_frame.clamp_min(fmin)
    log_f0 = torch.log(pitch_f_safe)  # [B, Tf]

    # ------------------------------------------------
    # Step 2: Reference Frame -> Reference Phoneme Mean
    # ------------------------------------------------
    # Map frame-level log-F0 to phoneme-level using ground truth alignment
    w_gd = attn_f2p_gd.transpose(1, 2).to(log_f0.dtype)  # [B, Tf, Tp]
    denom_gd = w_gd.sum(dim=1).clamp_min(1.0)  # [B, Tp]
    log_f0_p = torch.einsum("bt,btp->bp", log_f0, w_gd) / (denom_gd + eps)  # [B, Tp]

    # ------------------------------------------------
    # Step 3 & 4: Logic Branching based on attn_f2p_pred
    # ------------------------------------------------
    if attn_f2p_pred is None:
        # CASE 1: Training / Same-length reference
        # Broadcast reference phoneme means back to reference frames
        log_f0_back = torch.einsum("bp,btp->bt", log_f0_p, w_gd)  # [B, Tf]
        current_voiced_mask = voiced
    else:
        # CASE 2: Inference / Style Transfer (Cross-length)
        # Tp_syn is the number of phonemes in the target text
        Tp_syn = attn_f2p_pred.size(1)
        Tf_syn = attn_f2p_pred.size(2)

        # 3a) Interpolate phoneme-level pitch from Tp -> Tp_syn
        # We use linear interpolation to stretch/squeeze the pitch contour across phonemes
        log_f0_p_syn = F.interpolate(
            log_f0_p.unsqueeze(1),
            size=Tp_syn,
            mode="linear",
            align_corners=True
        ).squeeze(1)  # [B, Tp_syn]

        # 4a) Broadcast interpolated phoneme means to target frames
        w_pred = attn_f2p_pred.transpose(1, 2).to(log_f0.dtype)  # [B, Tf_syn, Tp_syn]
        log_f0_back = torch.einsum("bp,btp->bt", log_f0_p_syn, w_pred)  # [B, Tf_syn]

        # In inference, we don't have a frame-level voiced mask for target frames,
        # so we rely on Step 4's gap filling to handle any zeros created by broadcast.
        current_voiced_mask = (log_f0_back > 0)

        # ------------------------------------------------
    # Step 5: Interpolate Unvoiced Gaps
    # ------------------------------------------------
    log_f0_back = log_f0_back.unsqueeze(1)  # [B, 1, T]
    log_f0_filled = log_f0_back.clone()

    # Fill gaps to create a continuous "Trend" line
    log_f0_filled[:, 0, :] = _interp_unvoiced_1d(
        log_f0_back[:, 0, :], current_voiced_mask
    )

    # ------------------------------------------------
    # Step 6: Final Resample to Target Bert length
    # ------------------------------------------------
    if log_f0_filled.size(-1) != tgt_frame_len:
        log_f0_filled = F.interpolate(
            log_f0_filled,
            size=tgt_frame_len,
            mode="linear",
            align_corners=True,
        )

    return log_f0_filled


def get_phone_range_by_cut_f2p_attn(f2p_attn):
    """
    get batch-level phone range, and sample-level phone range.
    f2p_attn: (B, Tp, Tf)
    Returns: phn_start [B], phn_end [B]
    """
    phoneme_mask = f2p_attn.sum(dim=-1) > 0  # [B, Tp]
    phn_start = torch.argmax(phoneme_mask.float(), dim=-1)

    Tp = phoneme_mask.size(-1)
    phn_end = (Tp - 1) - torch.argmax(phoneme_mask.flip(dims=[-1]).float(), dim=-1)

    batch_phn_start = torch.min(phn_start)
    batch_phn_end = torch.max(phn_end)
    return batch_phn_start, batch_phn_end, phn_start, phn_end


def get_cut_phonemes_by_cut_f2p_attn(phonemes, f2p_attn):
    """
    phonemes: (B, Tp) - Phoneme IDs
    f2p_attn: (B, Tp, Tf) - Monotonic attention

    Returns: (B, Tp) - Same size as input, but indices outside the
                      attention window are set to 0.
    """
    B, Tp = phonemes.shape
    phn_start, phn_end = get_phone_range_by_cut_f2p_attn(f2p_attn)

    # Create a grid of indices [1, Tp] -> [B, Tp]
    indices = torch.arange(Tp, device=phonemes.device).unsqueeze(0).expand(B, Tp)

    # Create the window mask: True if start <= index <= end
    # We use .unsqueeze(1) to ensure broadcasting works against the indices grid
    mask = (indices >= phn_start.unsqueeze(1)) & (indices <= phn_end.unsqueeze(1))

    # Apply mask: Keep original phoneme ID if in window, else 0
    # .long() converts the boolean mask to 1s and 0s
    return phonemes * mask.long()


def align_attention_to_zero(f2p_attn_pred):
    """
    f2p_attn_pred: (B, Tp, Tf)
    Returns: aligned_attn (B, new_Tp, new_Tf)
    """
    B, Tp, Tf = f2p_attn_pred.shape
    device = f2p_attn_pred.device

    cropped_samples = []
    max_p = 0
    max_f = 0

    for i in range(B):
        # Find all indices where attn is 1
        indices = torch.nonzero(f2p_attn_pred[i])

        if indices.numel() == 0:
            # Handle empty sample case
            cropped_samples.append(torch.zeros((1, 1), device=device))
            continue

        # Get boundaries
        s_p, s_f = indices.min(dim=0)[0]
        e_p, e_f = indices.max(dim=0)[0] + 1

        # Crop the active region
        crop = f2p_attn_pred[i, s_p:e_p, s_f:e_f]
        cropped_samples.append(crop)

        # Update global max for the new batch shape
        max_p = max(max_p, crop.size(0))
        max_f = max(max_f, crop.size(1))

    # Create new aligned batch tensor
    aligned_attn = torch.zeros((B, max_p, max_f), device=device, dtype=f2p_attn_pred.dtype)

    for i, crop in enumerate(cropped_samples):
        p_len, f_len = crop.shape
        aligned_attn[i, :p_len, :f_len] = crop

    return aligned_attn


def print_gpu_memory(tag=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{tag}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB")

if __name__ == '__main__':
    B, Tp = 2, 6
    Tf = 5

    attn = torch.tensor([[
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
        [[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        ]], dtype=torch.int64)

    uv_mask = torch.tensor([
        [1, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1],
        ])

    cut_uv_mask_gd = torch.tensor([[
        0, 0, 1, 1, 0, 0
    ]])

    cut_uv_mask = get_phone_range_by_cut_f2p_attn(attn)
    print(cut_uv_mask)

    """
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
    pass
