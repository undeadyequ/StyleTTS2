import os
import shutil
import librosa
import soundfile as sf
import numpy as np

def process_wav_directory(input_dir, trim_wav_dir, trimmed_dir, top_db=25):
    """
    Detects tails, copies problematic files, and saves trimmed versions.
    """
    # Create output directories if they don't exist
    os.makedirs(trim_wav_dir, exist_ok=True)
    os.makedirs(trimmed_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} files. Starting detection...")

    for filename in wav_files:
        path = os.path.join(input_dir, filename)

        # 1. Load audio
        y, sr = librosa.load(path, sr=None)

        # 2. Detect if a trim is needed
        # librosa.effects.trim returns the indices of the non-silent interval
        yt, index = librosa.effects.trim(y, top_db=top_db)

        # Check if the 'tail' (end of file) was cut by more than 0.05s (approx 1000 samples @ 22k)
        # This prevents copying files that only have a few samples of silence.
        has_tail = (len(y) - index[1]) > (sr * 0.05)

        if has_tail:
            print(f"[DETECTED] {filename}: Tail of {len(y) - index[1]} samples found.")

            # 3. Copy original to "trim_wav" folder
            shutil.copy2(path, os.path.join(trim_wav_dir, filename))

            # 4. Save trimmed version to "trimmed" folder
            trimmed_path = os.path.join(trimmed_dir, filename)
            sf.write(trimmed_path, yt, sr)
        else:
            # Optional: handle files that are already clean
            pass

    print("\nProcessing complete.")
    print(f"Originals with tails copied to: {trim_wav_dir}")
    print(f"Cleaned versions saved to: {trimmed_dir}")


# Usage
process_wav_directory(
    input_dir='/home/rosen/ckpt/exp2/benchmark_esd/decodit_cfm_v34/random',
    trim_wav_dir='/home/rosen/ckpt/exp2/benchmark_esd/decodit_cfm_v34/random_tail',
    trimmed_dir='/home/rosen/ckpt/exp2/benchmark_esd/decodit_cfm_v34/random_trainTrimmed',
    top_db=25  # Adjust based on your DeCoDiT-TTS noise floor
)