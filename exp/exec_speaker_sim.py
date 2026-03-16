"""Speaker similarity evaluation: SIM-O and SIM-R.

SIM-O: cosine similarity between synthesized speech and the original reference wav.
SIM-R: cosine similarity between synthesized speech and the reference wav passed
       through the HiFi-GAN vocoder (domain-matched comparison).

Uses WavLM-Large via s3prl directly (bypasses UniSpeech to avoid 'models' namespace
conflict with StyleTTS2's models.py).
File pairing: spk0013_Angry_ref0_syn0.wav <-> spk0013_Angry_ref0.wav
"""

import os
import re
import json
import tempfile
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa

# Cache for loaded speaker encoder models
_spk_encoder_cache = {}


def _load_spk_encoder(model_name: str = 'wavlm_large', device=None):
    """Load and cache an s3prl upstream speaker encoder model."""
    import s3prl.hub as hub
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    key = (model_name, str(device))
    if key not in _spk_encoder_cache:
        model = getattr(hub, model_name)()
        model = model.to(device).eval()
        _spk_encoder_cache[key] = model
    return _spk_encoder_cache[key]


@torch.no_grad()
def _get_embedding(model, wav_path: str, device) -> torch.Tensor:
    """Extract mean-pooled, L2-normalized speaker embedding from a wav file."""
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0).to(device)  # (T,)
    output = model([wav])
    # last_hidden_state: (1, T', D) — mean-pool over time
    emb = output['last_hidden_state'].mean(dim=1)  # (1, D)
    return F.normalize(emb, dim=-1)


def compute_speaker_sim(wav1: str, wav2: str, model_name: str = 'wavlm_large',
                        use_gpu: bool = True) -> float:
    """Compute cosine speaker similarity between two wav files.

    Uses WavLM-Large (or other s3prl upstream models) directly via s3prl.hub,
    avoiding the UniSpeech 'models' namespace conflict with StyleTTS2's models.py.

    Args:
        wav1: Path to first wav file.
        wav2: Path to second wav file.
        model_name: s3prl upstream model name (default: 'wavlm_large').
        use_gpu: Use CUDA if available.

    Returns:
        Cosine similarity score (range -1 to 1; ~0.75+ for same speaker).
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = _load_spk_encoder(model_name, device)
    emb1 = _get_embedding(model, wav1, device)
    emb2 = _get_embedding(model, wav2, device)
    return float((emb1 * emb2).sum())


def vocode_wav(wav_path: str, vocoder, to_mel, device,
               mel_mean: float = -4, mel_std: float = 4) -> np.ndarray:
    """Pass a reference wav through the HiFi-GAN vocoder (for SIM-R).

    Computes the same normalized log-mel used during TTS synthesis, then
    runs the vocoder to produce a domain-matched audio signal.

    Args:
        wav_path: Path to source wav file.
        vocoder: Loaded HiFi-GAN generator (from load_vocoder.get_vocoder).
        to_mel: torchaudio.transforms.MelSpectrogram instance.
        device: torch device.
        mel_mean: Log-mel mean for normalization (default -4).
        mel_std: Log-mel std for normalization (default 4).

    Returns:
        Vocoded audio as numpy float32 array at 24 kHz.
    """
    wave, _ = librosa.load(wav_path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    wave_tensor = torch.from_numpy(audio).float()
    mel = to_mel(wave_tensor)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - mel_mean) / mel_std
    mel = mel.to(device)
    with torch.no_grad():
        out = vocoder(mel).squeeze().cpu().numpy()
    return out.astype(np.float32)


def compute_sim_for_dir(syn_dir, ref_dir, vocoder, to_mel, output_json,
                        model_name: str = 'wavlm_large', device=None,
                        need_syn_recon: bool = False) -> dict:
    """Compute SIM-O and SIM-R for all synthesized wavs in a directory.

    For each file like spk0013_Angry_ref0_syn0.wav in syn_dir, finds the
    corresponding reference spk0013_Angry_ref0.wav in ref_dir.

    Vocodes each unique reference once (cached) for SIM-R.

    Args:
        syn_dir: Directory of synthesized wavs.
        ref_dir: Directory of reference wavs.
        vocoder: Loaded HiFi-GAN generator.
        to_mel: torchaudio MelSpectrogram transform.
        output_json: Path to save per-pair results JSON.
        model_name: Speaker encoder model name.
        device: torch device.
        need_syn_recon: If True, SIM-R compares vocoded synthesized speech vs
                        vocoded reference (both domain-matched). If False
                        (default), SIM-R compares original synthesized speech
                        vs vocoded reference.

    Returns:
        dict with keys: per_pair, sim_o_mean, sim_o_std, sim_r_mean, sim_r_std
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    use_gpu = (str(device) != 'cpu')

    syn_wavs = sorted(glob.glob(os.path.join(str(syn_dir), '*.wav')))
    if len(syn_wavs) == 0:
        print(f"  ⚠ No wav files found in {syn_dir}")
        return {"per_pair": [], "sim_o_mean": float('nan'), "sim_o_std": float('nan'),
                "sim_r_mean": float('nan'), "sim_r_std": float('nan')}

    vocoded_cache = {}   # path (str) -> temp_wav_path (str)
    tmp_files = []
    results = []

    print(f"  Processing {len(syn_wavs)} files...")

    for i, syn_path in enumerate(syn_wavs):
        syn_name = os.path.basename(syn_path)
        # Strip _synN suffix to get reference filename
        ref_name = re.sub(r'_syn\d+', '', syn_name)
        ref_path = os.path.join(str(ref_dir), ref_name)

        if not os.path.exists(ref_path):
            print(f"  ⚠ Reference not found for {syn_name} (expected {ref_name}), skipping")
            continue

        try:
            # SIM-O: synthesized vs original reference
            sim_o = compute_speaker_sim(syn_path, ref_path, model_name, use_gpu)

            # SIM-R: vocoded reference (always); optionally also vocode synthesized
            if ref_path not in vocoded_cache:
                vocoded_audio = vocode_wav(ref_path, vocoder, to_mel, device)
                tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp.close()
                torchaudio.save(tmp.name,
                                torch.from_numpy(vocoded_audio).unsqueeze(0), 24000)
                vocoded_cache[ref_path] = tmp.name
                tmp_files.append(tmp.name)

            if need_syn_recon:
                # Vocode synthesized speech too (cached per syn_path)
                if syn_path not in vocoded_cache:
                    vocoded_syn = vocode_wav(syn_path, vocoder, to_mel, device)
                    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    tmp.close()
                    torchaudio.save(tmp.name,
                                    torch.from_numpy(vocoded_syn).unsqueeze(0), 24000)
                    vocoded_cache[syn_path] = tmp.name
                    tmp_files.append(tmp.name)
                sim_r = compute_speaker_sim(vocoded_cache[syn_path], vocoded_cache[ref_path],
                                            model_name, use_gpu)
            else:
                sim_r = compute_speaker_sim(syn_path, vocoded_cache[ref_path], model_name, use_gpu)

            results.append({
                "synthesized": syn_name,
                "reference": ref_name,
                "sim_o": round(float(sim_o), 6),
                "sim_r": round(float(sim_r), 6),
            })
        except Exception as e:
            print(f"  ⚠ Failed {syn_name}: {e}")

        if (i + 1) % 20 == 0 and results:
            print(f"  [{i+1}/{len(syn_wavs)}] latest SIM-O={results[-1]['sim_o']:.4f}, "
                  f"SIM-R={results[-1]['sim_r']:.4f}")

    # Cleanup temp files
    for tmp_path in tmp_files:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if len(results) == 0:
        output = {"per_pair": [], "sim_o_mean": float('nan'), "sim_o_std": float('nan'),
                  "sim_r_mean": float('nan'), "sim_r_std": float('nan')}
    else:
        sim_o_vals = np.array([r["sim_o"] for r in results])
        sim_r_vals = np.array([r["sim_r"] for r in results])
        output = {
            "per_pair": results,
            "sim_o_mean": round(float(sim_o_vals.mean()), 6),
            "sim_o_std":  round(float(sim_o_vals.std()),  6),
            "sim_r_mean": round(float(sim_r_vals.mean()), 6),
            "sim_r_std":  round(float(sim_r_vals.std()),  6),
        }

    if output_json is not None:
        os.makedirs(os.path.dirname(str(output_json)), exist_ok=True)
        with open(str(output_json), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    return output
