import torch
import torch.nn.functional as F
from math import pi, sin

def fuse_prosody_simple(pred_pitch, pred_energy, ref_pitch, ref_energy, voiced_mask, tau=0.15):
    """
    Simplified fusion gate:
    Reference dominates when voiced and stable.
    """
    eps = 1e-8
    device = pred_pitch.device
    T_pred = pred_pitch.shape[0]

    # --- Interpolate reference to predicted length ---
    def interp(x, target_len):
        """
        Linear interpolation that works for any PyTorch version.
        x: 1-D tensor [T_src]
        returns: 1-D tensor [target_len]
        """
        if len(x) == target_len:
            return x

        x = x.unsqueeze(0).unsqueeze(0)  # [1,1,T_src]
        x_resized = F.interpolate(
            x, size=target_len, mode="linear", align_corners=True
        )
        return x_resized.squeeze(0).squeeze(0)

    ref_pitch_i = interp(ref_pitch, T_pred)
    ref_energy_i = interp(ref_energy, T_pred)

    # --- 1. Stability of reference F0 ---
    logf0 = torch.log(ref_pitch_i + eps)
    dlogf0 = torch.abs(logf0 - torch.roll(logf0, 1))
    dlogf0[0] = dlogf0[1]
    c_stab = torch.exp(-dlogf0 / tau)

    # --- 2. Fusion weight α_t ---
    voiced = voiced_mask.to(device).float()  # from phoneme alignment
    alpha = voiced * c_stab                  # [0,1]

    # --- 3. Fuse pitch and energy ---
    fused_pitch  = alpha * ref_pitch_i + (1 - alpha) * pred_pitch
    fused_energy = alpha * ref_energy_i + (1 - alpha) * pred_energy

    return fused_pitch, fused_energy, alpha


def build_voiced_mask(phonemes_list, p2f_attn):
    """
    phonemes_list : list of phoneme strings, length Np
    p2f_attn      : tensor [Np, Nf] of {0,1} indicating phoneme→frame alignment
    returns       : voiced_mask [Nf] with 1 for voiced frames
    """
    # typical voiced phonemes in English
    voiced_phonemes = {
        "a","e","i","o","u",
        "aa","ae","ah","ao","aw","ay","eh","er","ey",
        "ih","iy","ow","oy","uh","uw",
        "b","d","g","v","z","zh","j","jh","l","m","n","ng","r","w","y"
    }

    # 1. phoneme-level voiced flag
    voiced_ph_flag = torch.tensor(
        [1.0 if ph in voiced_phonemes else 0.0 for ph in phonemes_list],
        device=p2f_attn.device
    )                                # [Np]

    # 2. project to frame level
    voiced_mask = voiced_ph_flag @ p2f_attn        # [Nf]
    voiced_mask = (voiced_mask > 0).float()        # clamp to {0,1}

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

    #
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
    fused_pitch, fused_energy, alpha = fuse_prosody_simple(
        pred_pitch, pred_energy, ref_pitch, ref_energy, voiced_mask, tau=0.15
    )

    print("alpha:", alpha.round(decimals=3))
    print("fused_pitch[:10]:", fused_pitch[:10])
    print("fused_energy[:10]:", fused_energy[:10])
