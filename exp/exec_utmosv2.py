#!/usr/bin/env python3
import os
import argparse
import glob
import torch
import random
import numpy as np
import torchaudio
import soundfile as sf
import utmosv2


def set_deterministic(seed: int = 42):
    """Make inference reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resample_folder_to_16k(src_dir: str, dst_dir: str):
    """Convert all wavs in src_dir to 16 kHz mono wavs in dst_dir."""
    wav_paths = sorted(glob.glob(os.path.join(src_dir, "**", "*.wav"), recursive=True))
    if not wav_paths:
        raise ValueError(f"No .wav files found in {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)
    out_paths = []

    for in_path in wav_paths:
        audio, sr = torchaudio.load(in_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            sr = 16000

        out_path = os.path.join(dst_dir, os.path.basename(in_path))
        sf.write(out_path, audio.squeeze(0).cpu().numpy(), sr, subtype="PCM_16")
        out_paths.append(out_path)

    return out_paths


def run_utmos(folder_16k: str, deterministic_seed: int = 42):
    """Run UTMOS-v2 on all wavs (no batch_size argument)."""
    set_deterministic(deterministic_seed)
    model = utmosv2.create_model(pretrained=True)
    model.eval()

    mos_list = model.predict(input_dir=folder_16k)
    scores = np.array([item["predicted_mos"] for item in mos_list], dtype=float)
    mean_mos = float(scores.mean())
    std_mos = float(scores.std())
    return mean_mos, std_mos, mos_list


def main():
    parser = argparse.ArgumentParser(description="Resample to 16 kHz and evaluate UTMOS-v2.")
    parser.add_argument("--inp", required=True, help="Folder containing input wavs (e.g., 24 kHz).")
    parser.add_argument("--out", required=True, help="Output folder for 16 kHz wavs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/2] Resampling {args.inp} → {args.out} (16 kHz mono)...")
    resampled_paths = resample_folder_to_16k(args.inp, args.out)
    print(f"   Converted {len(resampled_paths)} files.")

    print(f"[2/2] Evaluating UTMOS-v2 on {args.out} ...")
    mean_mos, std_mos, mos_list = run_utmos(args.out, args.seed)

    print(f"\nAverage UTMOS-v2: {mean_mos:.4f} ± {std_mos:.4f}")
    print(f"Files evaluated:", mos_list)


if __name__ == "__main__":
    main()
