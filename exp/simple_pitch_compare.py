import os
import librosa
import matplotlib.pyplot as plt

# Paths to your three folders
folders = ["folder1", "folder2", "folder3"]

# Four emotions (filenames must contain exactly these names)
emotions = ["angry.wav", "happy.wav", "sad.wav", "surprise.wav"]

def extract_pitch(path, sr=16000, fmin=50, fmax=500):
    """Extract pitch contour using librosa.pyin"""
    y, sr = librosa.load(path, sr=sr)
    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    # Replace NaN with 0 for plotting
    f0 = librosa.util.normalize(
        [0 if x is None else x for x in f0]
    )
    return f0

# Prepare plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, emotion in enumerate(emotions):
    ax = axes[i]
    for folder in folders:
        filepath = os.path.join(folder, emotion)
        if os.path.exists(filepath):
            f0 = extract_pitch(filepath)
            ax.plot(f0, label=os.path.basename(folder))
        else:
            print(f"Warning: {filepath} not found")
    ax.set_title(emotion.replace(".wav", "").capitalize())
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pitch (Hz)")
    ax.legend()

plt.tight_layout()
plt.show()