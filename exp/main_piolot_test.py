from extract_psd import extract_psdave, extract_psd_fine_class2, extract_psd
from exp.statcz_psd import statcz_psd_mcd, statcz_psd_fine_mcd, statcz_psd_mcd_fine_class2

"""
Evaluate temporal pitch/energy preservation with normalized DTW between
synthesized speech (modelA/modelB) and reference speech.

Inputs:
  --modelA_dir, --modelB_dir, --ref_dir

Outputs:
  - dtw_speech.csv
      rows: speech_id (e.g., spk0019_Surprise_ref0_syn0.wav)
      columns: modelA_dtw_pitch, modelA_dtw_energy, modelB_dtw_pitch, modelB_dtw_energy
  - dtw_model.csv
      rows: modelA, modelB
      columns: dtw_pitch, dtw_energy (averaged over all speeches)
  - prosody_contour_image/
      one image per speech_id, plotting pitch contours of modelA/modelB/reference
"""

import os
import re
import math
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


# -----------------------------
# DTW (normalized) with window
# -----------------------------
def dtw_distance_1d(x: np.ndarray, y: np.ndarray, window: Optional[int] = 50) -> Tuple[float, int]:
    """
    DTW for 1D sequences with optional Sakoe-Chiba window.
    Returns (total_cost, path_length).
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan, 0

    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    INF = np.float32(1e20)
    # cost DP
    dp = np.full((n + 1, m + 1), INF, dtype=np.float32)
    dp[0, 0] = 0.0

    # backpointer for path length
    prev = np.full((n + 1, m + 1, 2), -1, dtype=np.int32)
    prev[0, 0] = (0, 0)

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        xi = x[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(xi - y[j - 1])

            # choose best predecessor
            candidates = [
                (dp[i - 1, j], (i - 1, j)),      # insertion
                (dp[i, j - 1], (i, j - 1)),      # deletion
                (dp[i - 1, j - 1], (i - 1, j - 1))  # match
            ]
            best_val, best_p = min(candidates, key=lambda t: t[0])
            dp[i, j] = cost + best_val
            prev[i, j] = best_p

    total_cost = float(dp[n, m])

    # recover path length
    path_len = 0
    ci, cj = n, m
    if not np.isfinite(dp[n, m]):
        return np.nan, 0
    while not (ci == 0 and cj == 0):
        pi, pj = prev[ci, cj]
        if pi < 0 or pj < 0:
            break
        path_len += 1
        ci, cj = int(pi), int(pj)

    return total_cost, path_len


def normalized_dtw_1d(x: np.ndarray, y: np.ndarray, window: Optional[int] = 50) -> float:
    total, path_len = dtw_distance_1d(x, y, window=window)
    if not np.isfinite(total) or path_len <= 0:
        return np.nan
    return float(total) / float(path_len)


# -----------------------------
# Feature extraction
# -----------------------------
def extract_pitch_energy(
    wav_path: str,
    target_sr: int = 24000,
    hop_length: int = 300,
    fmin: float = 50.0,
    fmax: float = 600.0,
) -> Dict[str, np.ndarray]:
    """
    Pitch: librosa.pyin -> f0 (Hz) with voiced_flag
    Energy: RMS over frames (full contour)

    Returns:
      {
        "f0": (T,) float (NaN at unvoiced),
        "voiced": (T,) bool,
        "rms": (T,) float,
        "times": (T,) float seconds
      }
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # pitch
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        center=True,
    )

    # energy
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 4, hop_length=hop_length, center=True)[0]

    # align lengths (pyin and rms can differ by 1 frame depending on padding)
    T = min(len(f0), len(rms))
    f0 = f0[:T]
    voiced_flag = voiced_flag[:T]
    rms = rms[:T]
    times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)

    return {"f0": f0, "voiced": voiced_flag.astype(bool), "rms": rms, "times": times}


# -----------------------------
# Naming / pairing logic
# -----------------------------
_SYN_RE = re.compile(r"(.*)_syn(\d+)\.wav$", re.IGNORECASE)


def speech_id_to_ref_name(speech_id: str) -> Optional[str]:
    """
    "spk0019_Surprise_ref0_syn0.wav" -> "spk0019_Surprise_ref0.wav"
    """
    m = _SYN_RE.match(speech_id)
    if not m:
        return None
    return f"{m.group(1)}.wav"


# -----------------------------
# Plotting
# -----------------------------
def save_pitch_plot(
    out_path: str,
    speech_id: str,
    ref_feat: Dict[str, np.ndarray],
    a_feat: Dict[str, np.ndarray],
    b_feat: Dict[str, np.ndarray],
    modelA_name: str = "modelA",
    modelB_name: str = "modelB",
):
    plt.figure(figsize=(9.0, 3.2))
    plt.title(f"Pitch contours: {speech_id}", fontsize=12)

    # Use NaNs to create gaps for unvoiced frames (continuous line on voiced regions)
    plt.plot(ref_feat["times"], ref_feat["f0"], linewidth=2.5, label="Reference")
    plt.plot(a_feat["times"], a_feat["f0"], linewidth=1.6, label=modelA_name)
    plt.plot(b_feat["times"], b_feat["f0"], linewidth=1.6, label=modelB_name)

    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def get_model_name(path):
    path = os.path.normpath(path)
    return os.path.basename(os.path.dirname(path))


def _voiced_only_times_f0(feat: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    f0 = feat["f0"]
    t = feat["times"]
    v = feat["voiced"] & np.isfinite(f0)
    return t[v], f0[v]

def save_pitch_plot_voiced_only(
    out_path: str,
    speech_id: str,
    ref_feat: Dict[str, np.ndarray],
    a_feat: Dict[str, np.ndarray],
    b_feat: Dict[str, np.ndarray],
    modelA_name: str = "modelA",
    modelB_name: str = "modelB",
):
    plt.figure(figsize=(9.0, 3.2))
    plt.title(f"Voiced-only pitch contours: {speech_id}", fontsize=12)

    rt, rf0 = _voiced_only_times_f0(ref_feat)
    at, af0 = _voiced_only_times_f0(a_feat)
    bt, bf0 = _voiced_only_times_f0(b_feat)

    # Plot only voiced frames (unvoiced frames removed)
    plt.plot(rt, rf0, linewidth=2.5, label="Reference")
    plt.plot(at, af0, linewidth=1.6, label=modelA_name)
    plt.plot(bt, bf0, linewidth=1.6, label=modelB_name)

    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    speech_dir = "/home/rosen/Project/StyleTTS2/res/piolot_test"
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelA_dir", type=str, required=False, default=f"{speech_dir}/monoDiTmono/sharpLastPreserve_alpha0.3_beta0.7_mono02")
    ap.add_argument("--modelB_dir", type=str, required=False, default=f"{speech_dir}/monoDiT/sharpLastPreserve_alpha0.3_beta0.7")
    ap.add_argument("--ref_dir", type=str, required=False, default=f"{speech_dir}/monoDiT/reference_esd")
    ap.add_argument("--out_dir", type=str, default=f"{speech_dir}/sharpLastPreserve_alpha0.3_beta0.7_result_mono02")

    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--hop_length", type=int, default=300)
    ap.add_argument("--fmin", type=float, default=50.0)
    ap.add_argument("--fmax", type=float, default=600.0)

    # DTW window: lower is faster / stricter; None for full DTW
    ap.add_argument("--dtw_window", type=int, default=80)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "prosody_contour_image")
    os.makedirs(img_dir, exist_ok=True)

    modelA_name = get_model_name(args.modelA_dir)
    modelB_name = get_model_name(args.modelB_dir)

    # list wavs from modelA (canonical set)
    speech_ids = sorted([f for f in os.listdir(args.modelA_dir) if f.lower().endswith(".wav")])

    rows = []
    for sid in speech_ids:
        a_path = os.path.join(args.modelA_dir, sid)
        b_path = os.path.join(args.modelB_dir, sid)
        ref_name = speech_id_to_ref_name(sid)
        if ref_name is None:
            print(f"[Skip] Not matching '*_syn*.wav' pattern: {sid}")
            continue
        ref_path = os.path.join(args.ref_dir, ref_name)

        if not os.path.isfile(a_path):
            print(f"[Skip] Missing in modelA: {a_path}")
            continue
        if not os.path.isfile(b_path):
            print(f"[Skip] Missing in modelB: {b_path}")
            continue
        if not os.path.isfile(ref_path):
            print(f"[Skip] Missing in reference: {ref_path}")
            continue

        # extract features
        ref_feat = extract_pitch_energy(ref_path, args.sr, args.hop_length, args.fmin, args.fmax)
        a_feat = extract_pitch_energy(a_path, args.sr, args.hop_length, args.fmin, args.fmax)
        b_feat = extract_pitch_energy(b_path, args.sr, args.hop_length, args.fmin, args.fmax)

        # Pitch DTW on voiced-only contours
        ref_f0_v = ref_feat["f0"][ref_feat["voiced"] & np.isfinite(ref_feat["f0"])]
        a_f0_v = a_feat["f0"][a_feat["voiced"] & np.isfinite(a_feat["f0"])]
        b_f0_v = b_feat["f0"][b_feat["voiced"] & np.isfinite(b_feat["f0"])]

        dtw_pitch_a = normalized_dtw_1d(a_f0_v, ref_f0_v, window=args.dtw_window)
        dtw_pitch_b = normalized_dtw_1d(b_f0_v, ref_f0_v, window=args.dtw_window)

        # Energy DTW on full contours
        dtw_energy_a = normalized_dtw_1d(a_feat["rms"], ref_feat["rms"], window=args.dtw_window)
        dtw_energy_b = normalized_dtw_1d(b_feat["rms"], ref_feat["rms"], window=args.dtw_window)

        rows.append(
            {
                "speech_id": sid,
                f"{modelA_name}_dtw_pitch": dtw_pitch_a,
                f"{modelA_name}_dtw_energy": dtw_energy_a,
                f"{modelB_name}_dtw_pitch": dtw_pitch_b,
                f"{modelB_name}_dtw_energy": dtw_energy_b,
            }
        )

        # save pitch contour image
        out_img = os.path.join(img_dir, sid.replace(".wav", ".png"))
        save_pitch_plot_voiced_only(
            out_img,
            sid,
            ref_feat=ref_feat,
            a_feat=a_feat,
            b_feat=b_feat,
            modelA_name=modelA_name,
            modelB_name=modelB_name,
        )

    # dtw_speech.csv
    df_speech = pd.DataFrame(rows).set_index("speech_id").sort_index()
    speech_csv = os.path.join(args.out_dir, "dtw_speech.csv")
    df_speech.to_csv(speech_csv, float_format="%.6f")
    print(f"[Saved] {speech_csv}")

    # dtw_model.csv (averages over all speeches)
    def nanmean(series: pd.Series) -> float:
        v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        return float(np.nanmean(v)) if np.isfinite(np.nanmean(v)) else np.nan

    model_rows = []
    for model_name in [modelA_name, modelB_name]:
        model_rows.append(
            {
                "model": model_name,
                "dtw_pitch": nanmean(df_speech[f"{model_name}_dtw_pitch"]),
                "dtw_energy": nanmean(df_speech[f"{model_name}_dtw_energy"]),
            }
        )

    df_model = pd.DataFrame(model_rows).set_index("model")
    model_csv = os.path.join(args.out_dir, "dtw_model.csv")
    df_model.to_csv(model_csv, float_format="%.6f")
    print(f"[Saved] {model_csv}")

    print(f"[Saved] pitch contour images -> {img_dir}")
    print(f"[Done] evaluated {len(df_speech)} utterances.")


if __name__ == "__main__":
    main()
    #
    """
    prosody_dict = extract_psd(mel_config, out_speech_ablation_dir, model_n=ablation_name, save_psd_file="",
                               prosody_dict=prosody_dict)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}

    with open(psd_json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))

    psd_ctw_res, psd_mcd_stat_res, psd_mean_stat_res = statcz_psd_mcd(prosody_dict,
                                                                      exclude_zero=True)  # {"spk": {"ang": {"modelA": [p1, e1, m1]}, "happy":{ "modelB": []]}}}

    with open(psd_statics_mean_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(psd_mean_stat_res, sort_keys=True, indent=4))
    """