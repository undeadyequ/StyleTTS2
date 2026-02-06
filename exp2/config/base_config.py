"""Base configuration classes for experiments."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path


@dataclass
class PathConfig:
    """Centralized path management."""
    root_dir: Path
    output_dir: Path
    ref_dir: Path

    @classmethod
    def from_dataset(cls, dataset_name: str, exp_name: str):
        """Create path config from dataset name and experiment name."""
        root = Path(f"/home/rosen/ckpt/exp2/{exp_name}_{dataset_name}")
        return cls(
            root_dir=root,
            output_dir=root,
            ref_dir=root / "reference" / "random"
        )


@dataclass
class ExperimentConfig:
    """Base experiment configuration."""
    name: str
    dataset: str
    models: List[str]
    start_step: int = 0
    end_step: int = 6
    seed: int = 0

    # Feature flags
    save_attn: bool = False
    save_cond: bool = False
    save_attn_json: bool = True

    # PSD settings
    psd_level: str = "frame"  # "frame" or "phoneme"

    # Style/text naming
    style_syntex_name: str = "random"
    exp_name: str = "mdit_tts"

    def validate(self):
        """Validate configuration."""
        assert self.psd_level in ["frame", "phoneme"], f"Invalid psd_level: {self.psd_level}"
        assert 0 <= self.start_step <= self.end_step <= 6, "Invalid step range"
        assert self.dataset in ["esd", "libritts"], f"Invalid dataset: {self.dataset}"


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    show_spk: str
    show_emo: str
    show_txt: Tuple[int, int]
    show_t: int = 0
    show_h: int = 0
    tick_gran: str = "syllable"
