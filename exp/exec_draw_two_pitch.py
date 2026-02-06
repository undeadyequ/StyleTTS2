import numpy as np
import librosa
import matplotlib.pyplot as plt
import os


import numpy as np
import librosa
import matplotlib.pyplot as plt

# -------------------------
# 1) Pitch extraction (WORLD preferred)
# -------------------------
def extract_f0(wav_path, sr=24000, hop_length=300, f0_floor=50.0, f0_ceil=600.0):
    """
    Returns:
      t: (T,) time in seconds
      f0: (T,) f0 in Hz, unvoiced frames are 0.0
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # WORLD
    try:
        import pyworld as pw
        frame_period = hop_length / sr * 1000.0  # ms
        f0, t = pw.dio(
            y.astype(np.float64),
            fs=sr,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            frame_period=frame_period,
        )
        f0 = pw.stonemask(y.astype(np.float64), f0, t, sr)
        f0 = f0.astype(np.float64)

        # CHANGED: Use 0.0 instead of np.nan
        f0[f0 <= 0] = 0.0
        return t, f0

    except Exception:
        # Fallback: librosa.yin
        f0 = librosa.yin(
            y=y,
            fmin=f0_floor,
            fmax=f0_ceil,
            sr=sr,
            hop_length=hop_length,
        ).astype(np.float64)

        # Energy gate -> mark silence as unvoiced
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        if len(rms) < len(f0):
            rms = np.pad(rms, (0, len(f0) - len(rms)), mode="edge")
        else:
            rms = rms[:len(f0)]

        thr = np.percentile(rms, 20)

        # CHANGED: Use 0.0 instead of np.nan
        f0[rms < thr] = 0.0

        t = np.arange(len(f0)) * hop_length / sr
        return t, f0


def extract_energy_spectral(wav_path, sr=24000, hop_length=300, n_fft=2048, win_length=None, center=True,
                            log_scale=True):
    """
    Extracts spectral energy (norm of magnitude) to match PitEngExtractor.
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # Use win_length if provided, else default to n_fft
    win = win_length if win_length is not None else n_fft

    # 1. Get Linear Spectrogram (Magnitude)
    # Equivalent to your class's LinearSpectrogram/torch.stft
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win,
        center=center,
        pad_mode='reflect'
    )
    magnitude = np.abs(stft)  # [Freq, Time]

    # 2. Calculate Energy (Norm across frequency bins)
    # Equivalent to torch.norm(magnitude, dim=1)
    energy = np.linalg.norm(magnitude, axis=0)

    # 3. Apply Log Scale if requested
    if log_scale:
        energy = np.log(energy + 1e-7)

    t = np.arange(len(energy)) * hop_length / sr
    return t, energy.astype(np.float64)


# -------------------------
# 2) Make F0 continuous (interpolate gaps + smooth)
# -------------------------
def make_f0_continuous(
    f0,
    max_gap_frames=15,   # fill only short gaps by interpolation; longer gaps -> hold
    do_smooth=True,
    medfilt_kernel=5,
    savgol_window=9,
    savgol_poly=2,
):
    f0 = np.asarray(f0, dtype=np.float64).copy()
    f0[f0 <= 0] = np.nan

    n = len(f0)
    x = np.arange(n)
    voiced = np.isfinite(f0)

    if voiced.sum() == 0:
        return np.full_like(f0, np.nan)

    # Linear interpolation across NaNs -> fully continuous
    f0_filled = f0.copy()
    f0_filled[~voiced] = np.interp(x[~voiced], x[voiced], f0[voiced])

    # If gap is too long, do NOT interpolate across it (less "fake"):
    # use nearest voiced value (piecewise constant) instead.
    if max_gap_frames is not None:
        nan_mask = ~voiced
        edges = np.diff(np.concatenate([[0], nan_mask.view(np.int8), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) > max_gap_frames:
                left = s - 1
                right = e
                left_val = f0[left] if left >= 0 and np.isfinite(f0[left]) else None
                right_val = f0[right] if right < n and np.isfinite(f0[right]) else None
                if left_val is not None:
                    f0_filled[s:e] = left_val
                elif right_val is not None:
                    f0_filled[s:e] = right_val

    if not do_smooth:
        return f0_filled

    # Smoothing (needs scipy). If scipy not installed, just return f0_filled.
    try:
        from scipy.signal import medfilt, savgol_filter

        k = int(medfilt_kernel)
        if k % 2 == 0:
            k += 1
        k = min(k, n if n % 2 == 1 else n - 1)
        k = max(k, 3)

        f0_med = medfilt(f0_filled, kernel_size=k)

        w = int(savgol_window)
        if w % 2 == 0:
            w += 1
        w = min(w, n if n % 2 == 1 else n - 1)
        w = max(w, 5)

        f0_smooth = savgol_filter(f0_med, window_length=w, polyorder=savgol_poly)
        return f0_smooth

    except Exception:
        return f0_filled


# -------------------------
# 3) Camera-ready plot for your wav_paths + labels
# -------------------------
def plot_pitch_multi(
    wav_paths,
    labels,
    sr=24000,
    hop_length=300,
    f0_floor=50.0,
    f0_ceil=600.0,
    normalize_time=True,
    ylim=(50, 600),
    out_pdf="pitch_contours.pdf",
):
    assert len(wav_paths) == len(labels)

    # Camera-ready matplotlib config
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,  # editable text in PDF
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)

    # Plot each curve
    for wp, lab in zip(wav_paths, labels):
        t, f0 = extract_f0(wp, sr=sr, hop_length=hop_length, f0_floor=f0_floor, f0_ceil=f0_ceil)
        f0c = make_f0_continuous(
            f0,
            max_gap_frames=15,   # tune: 10-25
            do_smooth=True,
            medfilt_kernel=5,
            savgol_window=9,
            savgol_poly=2,
        )

        x = np.linspace(0.0, 1.0, len(f0c)) if normalize_time else t

        is_ref = (lab.lower() == "reference")
        ax.plot(
            x, f0c,
            linewidth=(3.2 if is_ref else 2.0),
            alpha=0.95,
            label=lab,
            zorder=(3 if is_ref else 2),
        )

    ax.set_title("Pitch Contours")
    ax.set_ylabel("F0 (Hz)")
    ax.set_xlabel("Normalized time" if normalize_time else "Time (s)")
    ax.set_ylim(*ylim)

    # Clean academic styling
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: compact, paper-friendly
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, ncol=2)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    ablation_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v4"
    ref_id, syn_id = 0, 1
    # cmp: ref0_syn1
    for ref_id in range(5):
        for syn_id in range(5):
            wav_paths = [
                os.path.join(ablation_dir, "monoDiT_ablation/none_m02_f02", f"spk0019_Angry_ref{ref_id}_syn{syn_id}.wav"),
                os.path.join(ablation_dir, "monoDiT_ablation/none_m05_f02", f"spk0019_Angry_ref{ref_id}_syn{syn_id}.wav"),
                os.path.join(ablation_dir, "monoDiT_ablation/none_m08_f02", f"spk0019_Angry_ref{ref_id}_syn{syn_id}.wav"),
                os.path.join(ablation_dir, "monoDiT_ablation/none_mm10_f02", f"spk0019_Angry_ref{ref_id}_syn{syn_id}.wav"),
                os.path.join(ablation_dir, "reference/random", f"spk0019_Angry_ref{ref_id}.wav"),
            ]
            labels = ["m02", "m05", "m08", "mm10", "reference"]
            plot_pitch_multi(
                wav_paths, labels,
                sr=24000, hop_length=300,
                f0_floor=50, f0_ceil=600,
                normalize_time=True,  # set False if they’re same length
                out_pdf=f"pitch_angry_all_ref{ref_id}_syn{syn_id}.png"
            )