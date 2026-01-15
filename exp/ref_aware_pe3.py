import torch
import torch.nn.functional as F

# --------------------------------------------------------
# Utility: smooth voiced mask
# --------------------------------------------------------
def smooth_voiced_mask(voiced_mask, kernel_size=9, sigma=1.5):
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, device=voiced_mask.device).float()
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    voiced_mask = voiced_mask.float().unsqueeze(0).unsqueeze(0)
    soft_mask = F.conv1d(voiced_mask, kernel[None, None, :], padding=half)
    return soft_mask.squeeze(0).squeeze(0).clamp(0, 1)

# --------------------------------------------------------
# Utility: interpolate unvoiced regions
# --------------------------------------------------------
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
            voiced_vals.cpu().numpy()
        )
    )
    return f0_interp_np.to(f0.device, dtype=f0.dtype)

# --------------------------------------------------------
# Utility: Gaussian smoothing
# --------------------------------------------------------
def smooth_pitch_gaussian(f0, sigma=1.0, kernel_size=9):
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, device=f0.device, dtype=f0.dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    f0_pad = F.pad(f0[None, None, :], (half, half), mode="reflect")
    smooth = F.conv1d(f0_pad, kernel[None, None, :])
    return smooth.view(-1)

# --------------------------------------------------------
# Utility: normalize voiced-only reference F0
# --------------------------------------------------------
def normalize_ref_pitch(ref_pitch, threshold=0.1):
    voiced = (ref_pitch > 1e-3)
    if voiced.sum() < 2:
        return torch.zeros_like(ref_pitch)
    v_vals = ref_pitch[voiced]
    min_v, max_v = v_vals.min(), v_vals.max()
    norm = 2 * (ref_pitch - min_v) / (max_v - min_v + 1e-8) - 1
    norm[~voiced] = 0
    norm[norm.abs() < threshold] = 0
    return norm.clamp(-1, 1)

# --------------------------------------------------------
# Main: Smooth additive fusion (Strategy 2)
# --------------------------------------------------------
def fuse_prosody_smooth_additive(
    pred_pitch, pred_energy,
    ref_pitch, ref_energy,
    voiced_mask,
    tau=0.15, beta=0.8, threshold=0.1,
    smooth_sigma=1.2, smooth_kernel=9,
    mask_kernel=9, mask_sigma=1.5,
    post_smooth=False,
    amplifier_min=0.2,
    amplifier_max=2,
):
    """
    Strategy 2:
    - Smooth voiced/unvoiced transition in voice mask.
    - Add normalized voiced reference bias to predicted F0.
    - Clean and smooth reference before normalization.
    """
    device = pred_pitch.device
    T_pred = pred_pitch.numel()

    # 1. Smooth the voiced/unvoiced transition (soft mask)
    voiced_soft = smooth_voiced_mask(voiced_mask.to(device), mask_kernel, mask_sigma)
    #print("voice_soft", voiced_soft, torch.max(voiced_soft), torch.min(voiced_soft))

    # 2. Compute stability from predicted F0
    eps = 1e-8
    logf0 = torch.log(pred_pitch.clamp(min=1e-5))
    dlogf0 = torch.abs(logf0 - torch.roll(logf0, 1))
    if T_pred > 1:
        dlogf0[0] = dlogf0[1]
    c_stab = torch.exp(-dlogf0 / tau)
    stable_voiced = (voiced_soft * c_stab).clamp(0.0, 1.0)
    #print("c_stab", c_stab, torch.max(c_stab))
    #print("stable_voiced", stable_voiced)

    # 3. Preprocess reference F0 (remove unvoiced noise, smooth, then normalize)
    voiced_ref = (ref_pitch > 1e-3).float()
    ref_pitch_filled = fill_unvoiced_with_interp(ref_pitch, voiced_ref)
    #print(ref_pitch, ref_pitch_filled)
    ref_pitch_smooth = smooth_pitch_gaussian(ref_pitch_filled, sigma=smooth_sigma, kernel_size=smooth_kernel)
    ref_pitch_smooth = ref_pitch_smooth * voiced_ref  # zero unvoiced again
    ref_pitch_norm = normalize_ref_pitch(ref_pitch_smooth, threshold)
    #print("ref_pitch_norm", ref_pitch_norm)

    # 4. Align reference to predicted length
    ref_pitch_i = F.interpolate(ref_pitch_norm[None, None, :], size=T_pred,
                                mode="linear", align_corners=True).squeeze()
    #print("ref_pitch_i", ref_pitch_i)

    # 5. Additive fusion
    #print("pred_pitch", pred_pitch)
    #mod = (1 + beta * stable_voiced * ref_pitch_i).clamp(amplifier_min, amplifier_max)
    mod = torch.exp(beta * stable_voiced * ref_pitch_i)
    fused_pitch = pred_pitch * mod
    fused_energy = pred_energy  # optional: add similar bias later

    # 6. Optional final smoothing
    if post_smooth:
        half = 3
        t = torch.arange(-half, half + 1, device=device).float()
        kernel = torch.exp(-0.5 * (t / 1.0) ** 2)
        kernel /= kernel.sum()
        f0_pad = F.pad(fused_pitch[None, None, :], (half, half), mode="reflect")
        fused_pitch = F.conv1d(f0_pad, kernel[None, None, :]).squeeze()

    return fused_pitch, fused_energy, stable_voiced
