import os
import torchaudio
import torch

def resample_wavs(root_dir, orig_sr=16000, target_sr=24000):
    """
    Recursively find all .wav files under root_dir and resample from orig_sr → target_sr.
    Overwrites original files safely.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                wav_path = os.path.join(dirpath, fname)
                print(f"Processing: {wav_path}")

                # Load audio
                try:
                    audio, sr = torchaudio.load(wav_path)
                except Exception as e:
                    print(f"  ❌ Failed to load: {e}")
                    continue

                # Skip non-16k files unless you want to force resample
                if sr != orig_sr:
                    print(f"  ⚠️ Skipping (sample rate = {sr}, expected {orig_sr})")
                    continue

                # Resample
                try:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=orig_sr,
                        new_freq=target_sr
                    )
                    audio_resampled = resampler(audio)
                    torchaudio.save(wav_path, audio_resampled, target_sr)
                    print("  ✅ Resampled & saved.")
                except Exception as e:
                    print(f"  ❌ Error during resample/save: {e}")

if __name__ == "__main__":
    root = "/home/rosen/ckpt/exp/mdit_tts_esd_fine1/drawspeech/05"  # <-- change me
    resample_wavs(root, orig_sr=16000, target_sr=24000)