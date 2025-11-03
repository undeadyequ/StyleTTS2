import torch
import torch.nn.functional as F
from math import pi, sin

# ------------------------------------------------------------
# Utility: 1D interpolation (for all PyTorch versions)
# ------------------------------------------------------------
def _interp_1d(x, target_len):
    if x.numel() == target_len:
        return x
    x = x.view(1, 1, -1)
    y = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
    return y.view(-1)


# ------------------------------------------------------------
# Utility: fill unvoiced frames by linear interpolation
# ------------------------------------------------------------

def fill_unvoiced_with_interp(f0: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
    """
    Linearly interpolate F0 only in voiced spans.
    Works in all PyTorch versions (no torch.interp dependency).

    Args:
        f0:     [T] reference F0 sequence (can contain zeros)
        voiced: [T] binary mask (1 = voiced, 0 = unvoiced)

    Returns:
        f0_interp: [T] with unvoiced frames linearly filled
    """
    idx = torch.arange(len(f0), device=f0.device, dtype=f0.dtype)
    voiced_idx = idx[voiced.bool()]
    voiced_vals = f0[voiced.bool()]

    # If not enough voiced frames to interpolate, just return as-is
    if len(voiced_idx) < 2:
        return f0.clone()

    # Convert to numpy for reliable interpolation, then back to torch
    # (because PyTorch <2.1 has no torch.interp)
    f0_interp_np = torch.from_numpy(
        __import__('numpy').interp(
            idx.cpu().numpy(),
            voiced_idx.cpu().numpy(),
            voiced_vals.cpu().numpy()
        )
    )

    return f0_interp_np.to(f0.device, dtype=f0.dtype)

# ------------------------------------------------------------
# Utility: Gaussian smoothing
# ------------------------------------------------------------
def smooth_pitch_gaussian(f0, sigma=1.0, kernel_size=9):
    """Gaussian 1-D smoothing of a 1D tensor."""
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, device=f0.device, dtype=f0.dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    f0_pad = F.pad(f0[None, None, :], (half, half), mode="reflect")
    smooth = F.conv1d(f0_pad, kernel[None, None, :])
    return smooth.view(-1)


# ------------------------------------------------------------
# Main fusion function
# ------------------------------------------------------------
def fuse_prosody_final(
    pred_pitch, pred_energy,
    ref_pitch, ref_energy,
    voiced_mask,          # from phoneme-frame alignment
    tau=0.15,
    smooth_sigma=1.2,
    smooth_kernel=9
):
    """
    Reference dominates in stable voiced regions.
    - Stability c_t computed from predicted F0
    - Reference pitch smoothed (voiced-only)
    - Unvoiced reference regions ignored

    Returns:
        fused_pitch, fused_energy, alpha
    """
    device = pred_pitch.device
    T_pred = pred_pitch.numel()

    # --------------------------------------------------------
    # 1. Remove unvoiced noise and smooth reference pitch
    # --------------------------------------------------------
    voiced_ref = (ref_pitch > 1e-3).float()
    ref_pitch_filled = fill_unvoiced_with_interp(ref_pitch, voiced_ref)
    ref_pitch_smooth = smooth_pitch_gaussian(
        ref_pitch_filled, sigma=smooth_sigma, kernel_size=smooth_kernel
    )
    ref_pitch_smooth = ref_pitch_smooth * voiced_ref  # zero unvoiced again

    # --------------------------------------------------------
    # 2. Interpolate reference to predicted length
    # --------------------------------------------------------
    ref_pitch_i  = _interp_1d(ref_pitch_smooth.to(device), T_pred)
    ref_energy_i = _interp_1d(ref_energy.to(device), T_pred)

    # --------------------------------------------------------
    # 3. Compute stability c_t from predicted F0
    # --------------------------------------------------------
    eps = 1e-8
    safe_pred_f0 = pred_pitch.clamp(min=1e-5)
    logf0 = torch.log(safe_pred_f0)
    dlogf0 = torch.abs(logf0 - torch.roll(logf0, 1))
    if T_pred > 1:
        dlogf0[0] = dlogf0[1]
    c_stab = torch.exp(-dlogf0 / tau)

    # --------------------------------------------------------
    # 4. α_t = voiced_mask * c_stab   (voiced from phoneme alignment)
    # --------------------------------------------------------
    voiced = voiced_mask.to(device).float().clamp(0, 1)
    alpha = (voiced * c_stab).clamp(0.0, 1.0)

    # --------------------------------------------------------
    # 5. Blend reference & predicted (voiced-only reference)
    # --------------------------------------------------------
    ref_pitch_i  = torch.where(voiced > 0, ref_pitch_i,  pred_pitch)
    ref_energy_i = torch.where(voiced > 0, ref_energy_i, pred_energy)

    fused_pitch  = alpha * ref_pitch_i  + (1 - alpha) * pred_pitch
    fused_energy = alpha * ref_energy_i + (1 - alpha) * pred_energy

    return fused_pitch, fused_energy, alpha

def build_voiced_mask(phonemes_list, p2f_attn):
    """
    Build voiced mask from IPA phonemes and phoneme→frame attention.

    Args:
        phonemes_list : list of IPA phoneme strings, length [N_p]
                        (e.g., ["s", "a", "m", "p", "l", "ɚ"])
        p2f_attn      : tensor [N_p, N_f] of {0,1} mapping phonemes→frames

    Returns:
        voiced_mask : tensor [N_f] with 1 for voiced frames
    """
    # ----------------------------
    # 1. Define voiced IPA symbols
    # ----------------------------
    voiced_ipa = {
        # vowels
        "i", "y", "ɨ", "ʉ", "ɯ", "u",
        "ɪ", "ʏ", "ʊ",
        "e", "ø", "ɘ", "ɵ", "ɤ", "o",
        "ə", "ɛ", "œ", "ɜ", "ɞ", "ʌ", "ɔ",
        "æ", "ɐ", "a", "ɶ", "ɑ", "ɒ",
        # voiced consonants
        "b", "d", "ɡ", "v", "ð", "z", "ʒ",
        "ʝ", "ɣ", "ʁ", "ʕ", "ɦ",
        "m", "n", "ŋ", "ɱ", "ɳ", "ɲ", "ŋ̊",
        "l", "ɫ", "ɭ", "ʎ", "r", "ɹ", "ɻ", "ɾ",
        "w", "j",
    }

    # ----------------------------
    # 2. Phoneme-level voiced flags
    # ----------------------------
    voiced_ph_flag = torch.tensor(
        [1.0 if ph in voiced_ipa else 0.0 for ph in phonemes_list],
        device=p2f_attn.device
    )  # [N_p]

    # ----------------------------
    # 3. Project to frame-level mask
    # ----------------------------
    # p2f_attn is [N_p, N_f] 0/1 matrix, summing across phonemes covering each frame
    voiced_mask = voiced_ph_flag @ p2f_attn     # [N_f]
    voiced_mask = (voiced_mask > 0).float()     # clamp to {0,1}

    return voiced_mask


if __name__ == '__main__':
    phonemes = ["s", "a", "m"]
    # each phoneme occupies 3,4,2 frames respectively
    p2f = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1]
    ], dtype=torch.float32)

    voiced_mask = build_voiced_mask(phonemes, p2f)
    print("voiced mask", voiced_mask)

    T_pred = 40  # length of predicted prosody
    T_ref = 55  # length of reference prosody (different length)

    # predicted (flat)
    pred_pitch = 0.3 * torch.sin(torch.linspace(0, 2 * pi, T_pred))
    pred_energy = 0.2 * torch.sin(torch.linspace(0, 3 * pi, T_pred))

    # reference (more expressive)
    ref_pitch = 1.5 * torch.sin(torch.linspace(0, 2 * pi, T_ref)) + 0.3 * torch.randn(T_ref)
    ref_energy = 1.2 * torch.sin(torch.linspace(0, 3 * pi, T_ref)) + 0.2 * torch.randn(T_ref)

    # clamp or scale into [-3, 3]
    for x in (pred_pitch, pred_energy, ref_pitch, ref_energy):
        x.clamp_(-3, 3)

    # voiced mask (simulating vowels/consonants)
    voiced_mask = torch.zeros(T_pred)
    voiced_mask[5:15] = 1  # first voiced region
    voiced_mask[22:35] = 1  # second voiced region

    # ---- Inspect results ----------------------------------------------------
    fused_pitch, fused_energy, alpha = fuse_prosody_final(
        pred_pitch, pred_energy, ref_pitch, ref_energy, voiced_mask, tau=0.15
    )

    print("alpha:", alpha.round(decimals=3)[:10])
    print("reference pitch: ", ref_pitch[:10])
    print("predicted pitch: ", pred_pitch[:10])
    print("fused_pitch[:10]:", fused_pitch[:10])
    print("fused_energy[:10]:", fused_energy[:10])
