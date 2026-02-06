"""Model configuration and registry."""

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING
from pathlib import Path
import sys
import os
import re

if TYPE_CHECKING:
    from exp2.config.ablation_config import AblationVariant

# Import model metadata from original const_param
sys.path.append('/home/rosen/Project/StyleTTS2')
sys.path.append('/home/rosen/Project/StyleTTS2/exp')
from exp.const_param import model_meta, model_infer_config


@dataclass
class ModelMetadata:
    """Metadata for a single model."""
    name: str
    root_dir: Path
    checkpoint: str
    config_file: str
    inference_params: Dict

    @property
    def checkpoint_path(self) -> Path:
        """Full path to checkpoint."""
        return self.root_dir / self.checkpoint

    @property
    def config_path(self) -> Path:
        """Full path to config file."""
        return self.root_dir / self.config_file


class ModelRegistry:
    """Central registry for all models."""

    _registry: Dict[str, ModelMetadata] = {}

    @classmethod
    def register(cls, name: str, root_dir: str, checkpoint: str, config_file: str, inference_params: Dict):
        """Register a model."""
        model = ModelMetadata(
            name=name,
            root_dir=Path(root_dir),
            checkpoint=checkpoint,
            config_file=config_file,
            inference_params=inference_params
        )
        cls._registry[name] = model

    @classmethod
    def get(cls, name: str) -> ModelMetadata:
        """Get model metadata by name."""
        if name not in cls._registry:
            raise ValueError(f"Model {name} not registered. Available models: {cls.list_models()}")
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._registry.keys())

    @classmethod
    def get_variant(cls, variant: 'AblationVariant') -> 'ModelMetadata':
        """Get model metadata with ablation overrides applied.

        Args:
            variant: AblationVariant specifying base model and overrides

        Returns:
            ModelMetadata with epoch and params adjusted
        """
        # Get base model
        base = cls.get(variant.base_model)

        # Create modified checkpoint path if epoch is specified
        if variant.epoch is not None:
            checkpoint = variant.get_checkpoint_path(
                base.root_dir,
                base.checkpoint
            )
            # Make path relative to root_dir
            checkpoint = str(checkpoint.relative_to(base.root_dir))
        else:
            checkpoint = base.checkpoint

        # Merge inference params with overrides
        merged_params = {**base.inference_params, **variant.param_overrides}

        return ModelMetadata(
            name=variant.variant_name,  # Use generated variant name
            root_dir=base.root_dir,
            checkpoint=checkpoint,
            config_file=base.config_file,
            inference_params=merged_params
        )


# Register models from const_param
def initialize_registry():
    """Initialize model registry with models from const_param."""
    for model_name, model_info in model_meta.items():
        # model_info is a list: [root_dir, checkpoint, config_file]
        # Some models (like hierspeech) may have only [root_dir, config_file]
        if len(model_info) == 3:
            root_dir, ckpt, config_file = model_info
        elif len(model_info) == 2:
            root_dir, config_file = model_info
            ckpt = ""  # No checkpoint for this model
        else:
            print(f"Warning: Skipping model {model_name} with unexpected format")
            continue

        infer_params = model_infer_config.get(model_name, {})
        ModelRegistry.register(
            name=model_name,
            root_dir=root_dir,
            checkpoint=ckpt,
            config_file=config_file,
            inference_params=infer_params
        )

# Auto-initialize on import
initialize_registry()
