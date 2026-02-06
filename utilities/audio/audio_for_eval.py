import torch
from matplotlib.mlab import magnitude_spectrum
from torch import Tensor
import torch.nn as nn
import torchaudio
import pyworld as pw
import numpy as np
import librosa
import os

class LinearSpectrogram(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, pad, center, pad_mode):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad
        self.center = center
        self.pad_mode = pad_mode
        
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        #waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (self.pad, self.pad), self.pad_mode).squeeze(1)
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (
                int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)), self.pad_mode).squeeze(1)

        spec = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length, self.window, self.center, self.pad_mode,
                          False, True, True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, f_min, f_max, pad, n_mels, center, pad_mode, mel_scale):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.center = center
        self.pad_mode = pad_mode
        self.mel_scale = mel_scale
        
        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, pad, center, pad_mode)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min, f_max, (n_fft//2)+1, mel_scale, mel_scale)

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def forward(self, x: Tensor) -> Tensor:
        linear_spec = self.spectrogram(x)
        x = self.mel_scale(linear_spec)
        x = self.compress(x)
        return x
    
def load_and_resample_audio(audio_path, target_sr, device='cpu') -> Tensor:
    try:
        y, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(str(e))
        return None
    
    y.to(device)
    # Convert to mono
    if y.size(0) > 1:
        y = y[0, :].unsqueeze(0) # shape: [2, time] -> [time] -> [1, time]
        
    # resample audio to target sample_rate
    if sr != target_sr:
        y = torchaudio.functional.resample(y, sr, target_sr)
    return y


def load_audio(audio_path, device='cpu') -> Tensor:
    try:
        y, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(str(e))
        return None
    y.to(device)
    # Convert to mono
    if y.size(0) > 1:
        y = y[0, :].unsqueeze(0)  # shape: [2, time] -> [time] -> [1, time]
    return y


class PitEngExtractor:
    def __init__(self, sample_rate, n_fft, win_length, hop_length, f_min, f_max, pad, n_mels, center, pad_mode, mel_scale, need_energy=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.center = center
        self.pad_mode = pad_mode
        self.mel_scale = mel_scale
        self.need_energy = need_energy
        if need_energy:
            self.linearspectrogram = LinearSpectrogram(n_fft, win_length, hop_length, pad, center, pad_mode)

    def forward(self, wav: torch.Tensor, need_rm_ood=False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        pitch >= energy + 1
        """
        if self.need_energy:
            magnitude = self.linearspectrogram(wav)
            energy = torch.norm(magnitude, dim=1).squeeze(0)
        else:
            energy = None
        if wav.ndim == 2:
            wav = wav.squeeze(0)
        wav_np = wav.numpy()  # [l, ]
        pitch, t = pw.dio(wav_np.astype(np.float64),
                          self.sample_rate, frame_period=self.hop_length / self.sample_rate * 1000)
        pitch = pw.stonemask(wav_np.astype(np.float64), pitch, t, self.sample_rate)
        if np.sum(pitch != 0) <= 1:
            return None
        if need_rm_ood:
            pitch = self.remove_outlier(pitch)
        return torch.from_numpy(pitch), energy

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        #normal_value = values[np.logical_and(values > lower, values < upper)]
        normal_value = np.clip(values, p25, p75)
        return normal_value

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

def normalize_pitch(pitch: Tensor, min_pitch: float = 50.0, max_pitch: float = 500.0) -> Tensor:
    """
    Normalize pitch values to a range between 0 and 1.
    
    Args:
        pitch (Tensor): (sample_size,1)
        min_pitch (float): Minimum pitch value.
        max_pitch (float): Maximum pitch value.
        
    Returns:
        Tensor: Normalized pitch values.
    """
    
    return (pitch - min_pitch) / (max_pitch - min_pitch)


def normalize_by_global_stats(pitch_values, global_mean, global_std):
    voiced_mask = pitch_values > 0
    normalized = torch.zeros_like(pitch_values)
    normalized[voiced_mask] = (pitch_values[voiced_mask] - global_mean) / (global_std + 1e-8)
    return normalized


if __name__ == '__main__':
    from exp.mel_config import MelConfig
    from dataclasses import dataclass, asdict
    from sklearn.preprocessing import StandardScaler
    from exp.vis2 import plot_f0_multi

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mel_config = MelConfig()

    mel_extractor = LogMelSpectrogram(**asdict(mel_config)).to(device)
    pitch_extractor = PitEngExtractor(**asdict(mel_config), need_energy=True)
    wav = "/home/rosen/Project/naturalspeech3_facodec/audio/1.wav"
    wav = "/home/rosen/data/ESD/0019/Surprise/train/0019_001457.wav"
    wav = "/hdd/LJSpeech/wavs/LJ001-0041.wav"
    #audio = load_and_resample_audio(wav, mel_config.sample_rate, device=device)  # shape: [1, time]
    #mel = mel_extractor(audio.to(device))
    #pit, eng = pitch_extractor.forward(audio, need_rm_ood=False)

    # multiple wavs from models
    root_dir = "/home/rosen/ckpt/exp/mdit_tts_esd"
    wav_dirs = ["styletts2", "monoDiT", "reference", "monoDiT_ab0808_m08_fbnone"]
    wav = "spk0019_Angry_ref1_syn1.wav"
    wav_ref = "spk0019_Angry_ref1.wav"
    wavs_paths = [os.path.join(root_dir, wav_dir, "random", wav) if wav_dir != "reference" else
                  os.path.join(root_dir, wav_dir, "random", wav_ref) for wav_dir in wav_dirs]
    pits = []
    for wav_path in wavs_paths:
        audio = load_and_resample_audio(wav_path, mel_config.sample_rate, device=device)  # shape: [1, time]
        pit, eng = pitch_extractor.forward(audio, need_rm_ood=False)
        pit = pit[pit > 0]
        pits.append(pit)

    plot_f0_multi(pits, model_names=wav_dirs, out_path="/home/rosen/Project/StyleTTS2/exp/res/styletts2_monoDiT_reference_noninterpolate_ang_ref1_syn1.png",
                  need_interpolate=False)

    #print(mel.shape, pit.shape)
    #print(pit)

    #pitch_mean, pitch_std = 127.06710720355903, 109.42779424391766
    #n_pit = normalize_by_global_stats(pit, pitch_mean, pitch_std)
    #print(n_pit)
    #pitch_scaler = StandardScaler()
    #pit_n = pit.reshape(-1, 1)
    #pitch_scaler.partial_fit(pit_n)
