from dataclasses import dataclass

@dataclass
class MelConfig:
    sample_rate: int = 24000  # 44100
    n_fft: int = 2048
    win_length: int = 1200
    hop_length: int = 300
    f_min: float = 0.0
    f_max: float = 8000.0
    pad: int = 0
    n_mels: int = 80
    center: bool = False
    pad_mode: str = "reflect"
    mel_scale: str = "slaney"

    def __post_init__(self):
        if self.pad == 0:
            self.pad = (self.n_fft - self.hop_length) // 2