import librosa
import soundfile as sf

def trim_synthesis_tail(input_path, output_path, top_db=25):
    """
    Trims the 'rising pitch' tail from synthesized speech by removing
    trailing silence/artifacts based on a decibel threshold.
    """
    # 1. Load the synthesized audio
    y, sr = librosa.load(input_path, sr=None)

    # 2. Find the non-silent interval
    # top_db=25 is a good starting point for TTS artifacts.
    # Lower it (e.g., 20) to be more aggressive, raise it (e.g., 30) to keep more tail.
    yt, index = librosa.effects.trim(y, top_db=top_db)

    # 3. Output trimming details for your research log
    duration_before = librosa.get_duration(y=y, sr=sr)
    duration_after = librosa.get_duration(y=yt, sr=sr)
    print(f"Trimmed: {duration_before:.3f}s -> {duration_after:.3f}s")

    # 4. Save the trimmed file
    sf.write(output_path, yt, sr)
    print(f"File saved to: {output_path}")


# Apply to your specific file
trim_synthesis_tail(
    input_path='/home/rosen/Project/StyleTTS2/exp/res/spk0013_Happy_ref3_syn1.wav',
    output_path='/home/rosen/Project/StyleTTS2/exp/res/spk0013_Happy_ref3_syn1_trimmed.wav'
)

