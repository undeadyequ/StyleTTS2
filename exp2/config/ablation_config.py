"""Ablation configuration for flexible model evaluation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import itertools
import re


@dataclass
class AblationVariant:
    """Represents a specific ablation configuration variant.

    Attributes:
        base_model: Base model name from const_param.py (e.g., "decodit_cfm_v29")
        epoch: Optional epoch number override (None = use base model's default)
        param_overrides: Inference parameter overrides (e.g., {"trend_strength": 3.0})
    """
    base_model: str
    epoch: Optional[int] = None
    param_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def variant_name(self) -> str:
        """Generate readable variant name for output directory.

        Example: decoditV29_epoch50_trendStren3
        """
        parts = [self._short_model_name()]

        if self.epoch is not None:
            parts.append(f"epoch{self.epoch}")

        # Add key parameter overrides to name (sorted for consistency)
        for key, value in sorted(self.param_overrides.items()):
            short_key = self._shorten_param_name(key)
            short_val = self._format_value(value)
            parts.append(f"{short_key}{short_val}")

        return "_".join(parts)

    def _short_model_name(self) -> str:
        """Shorten model name for directory.

        decodit_cfm_v29 -> decoditV29
        monoDiT -> monoDiT
        """
        name = self.base_model
        if "decodit_cfm_v" in name:
            version = name.split("_v")[-1]
            return f"decoditV{version}"
        if "monoDiT" in name:
            return "monoDiT"
        if "styletts2" in name:
            return "styletts2"
        return name.replace("_", "")

    def _shorten_param_name(self, key: str) -> str:
        """Shorten parameter names for directory."""
        mappings = {
            "trend_strength": "trendStren",
            "cfg_strength": "cfg",
            "diffusion_steps": "steps",
            "drop_trend": "dropTrd",
            "alpha": "a",
            "beta": "b",
            "hierStyle": "hier",
        }
        return mappings.get(key, key)

    def _format_value(self, value: Any) -> str:
        """Format value for directory name."""
        if isinstance(value, float):
            # 3.0 -> "3", 0.5 -> "0m5"
            if value == int(value):
                return str(int(value))
            return str(value).replace(".", "m")
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def get_checkpoint_path(self, root_dir: Path, checkpoint_template: str) -> Path:
        """Get checkpoint path with epoch substitution.

        Args:
            root_dir: Model root directory
            checkpoint_template: Template like "first_txt2mel_cfm_v29/epoch_2nd_00048.pth"

        Returns:
            Full checkpoint path with epoch substituted
        """
        if self.epoch is None:
            return root_dir / checkpoint_template

        # Replace epoch number in checkpoint path
        # epoch_2nd_00048.pth -> epoch_2nd_00050.pth
        new_ckpt = re.sub(
            r'epoch_2nd_(\d+)\.pth',
            f'epoch_2nd_{self.epoch:05d}.pth',
            checkpoint_template
        )
        return root_dir / new_ckpt


@dataclass
class AblationGrid:
    """Defines a grid of ablation variants to evaluate.

    Attributes:
        base_model: Base model name from const_param.py
        epochs: List of epoch numbers to evaluate (empty = use default)
        param_grid: Dict mapping parameter names to lists of values to try

    Example:
        grid = AblationGrid(
            base_model="decodit_cfm_v29",
            epochs=[48, 50],
            param_grid={"trend_strength": [3.0, 5.0]}
        )
        # Generates 4 variants: epoch48_trendStren3, epoch48_trendStren5, ...
    """
    base_model: str
    epochs: List[int] = field(default_factory=list)
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)

    def generate_variants(self) -> List[AblationVariant]:
        """Generate all combinations of ablation variants."""
        variants = []

        # Handle epochs (empty list means use default = [None])
        epoch_list = self.epochs if self.epochs else [None]

        # Handle param grid - generate all combinations
        if self.param_grid:
            param_keys = list(self.param_grid.keys())
            param_values = list(self.param_grid.values())
            param_combinations = list(itertools.product(*param_values))
        else:
            param_keys = []
            param_combinations = [()]  # Single empty combination

        # Generate all variants
        for epoch in epoch_list:
            for param_combo in param_combinations:
                overrides = dict(zip(param_keys, param_combo))
                variants.append(AblationVariant(
                    base_model=self.base_model,
                    epoch=epoch,
                    param_overrides=overrides
                ))

        return variants

    @classmethod
    def from_dict(cls, config: Dict) -> 'AblationGrid':
        """Create from dictionary (e.g., loaded from YAML or JSON)."""
        return cls(
            base_model=config["base_model"],
            epochs=config.get("epochs", []),
            param_grid=config.get("param_grid", {})
        )

    def __len__(self) -> int:
        """Return the number of variants that will be generated."""
        n_epochs = len(self.epochs) if self.epochs else 1
        n_params = 1
        for values in self.param_grid.values():
            n_params *= len(values)
        return n_epochs * n_params
