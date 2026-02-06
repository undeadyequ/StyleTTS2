"""Grid-based ablation study experiment."""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Any

sys.path.append('/home/rosen/Project/StyleTTS2')

from exp2.config.base_config import ExperimentConfig, PathConfig
from exp2.config.ablation_config import AblationGrid, AblationVariant
from exp2.pipeline.base_pipeline import Pipeline
from exp2.pipeline.synthesis_stage import SynthesisStage
from exp2.pipeline.extraction_stage import PSDExtractionStage
from exp2.pipeline.statistics_stage import StatisticsStage
from exp2.pipeline.quality_stage import QualityEvaluationStage


class GridAblationStudy:
    """Run ablation study over a grid of configurations.

    This allows evaluating the same base model with different epochs
    and inference parameters, with automatic variant naming.

    Example:
        grid = AblationGrid(
            base_model="decodit_cfm_v29",
            epochs=[48, 50],
            param_grid={"trend_strength": [3.0, 5.0]}
        )
        study = GridAblationStudy(config, grid)
        study.run(syn_styles, syn_texts, MelConfig)
    """

    def __init__(self, config: ExperimentConfig, ablation_grid: AblationGrid):
        self.config = config
        self.ablation_grid = ablation_grid
        self.variants = ablation_grid.generate_variants()
        self.paths = PathConfig.from_dataset(
            config.dataset,
            f"{config.exp_name}_ablation"
        )

    def run(self, syn_styles: List[Tuple], syn_texts: List[str], mel_config):
        """Execute ablation study pipeline."""

        # Print study overview
        print(f"\n{'='*60}")
        print(f"Grid Ablation Study: {self.ablation_grid.base_model}")
        print(f"{'='*60}")
        print(f"Dataset: {self.config.dataset}")
        print(f"Epochs: {self.ablation_grid.epochs or ['default']}")
        print(f"Parameter grid: {self.ablation_grid.param_grid or 'none'}")
        print(f"Total variants: {len(self.variants)}")
        print(f"\nVariants to evaluate:")
        for v in self.variants:
            print(f"  - {v.variant_name}")
        print(f"\nOutput: {self.paths.output_dir}")
        print(f"Steps: {self.config.start_step} -> {self.config.end_step}")
        print(f"{'='*60}\n")

        # Build pipeline with variants (pass AblationVariant objects directly)
        pipeline = Pipeline([
            SynthesisStage(self.variants),
            PSDExtractionStage(self.config.psd_level),
            StatisticsStage(),
            QualityEvaluationStage(),
        ])

        # Create context with variant names for downstream stages
        variant_names = [v.variant_name for v in self.variants]
        context = {
            "syn_styles": syn_styles,
            "syn_texts": syn_texts,
            "models": variant_names,  # Use variant names for stats/quality stages
            "output_dir": str(self.paths.output_dir),
            "style_syntex_name": self.config.style_syntex_name,
            "start_step": self.config.start_step,
            "end_step": self.config.end_step,
            "mel_config": mel_config,
            "save_attn": self.config.save_attn,
            "save_cond": self.config.save_cond,
            "save_attn_json": getattr(self.config, 'save_attn_json', False),
        }

        # Ensure output directory exists
        Path(self.paths.output_dir).mkdir(parents=True, exist_ok=True)

        # Execute pipeline
        results = pipeline.run(context)

        print(f"\n{'='*60}")
        print("Grid Ablation Study Complete!")
        print(f"Results saved to: {self.paths.output_dir}")
        print(f"{'='*60}\n")

        return results
