"""Random synthesis evaluation experiment."""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Any

sys.path.append('/home/rosen/Project/StyleTTS2')

from exp2.config.base_config import ExperimentConfig, PathConfig
from exp2.pipeline.base_pipeline import Pipeline
from exp2.pipeline.synthesis_stage import SynthesisStage
from exp2.pipeline.extraction_stage import PSDExtractionStage
from exp2.pipeline.statistics_stage import StatisticsStage
from exp2.pipeline.quality_stage import QualityEvaluationStage


class RandomEvaluation:
    """Standard random synthesis evaluation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.paths = PathConfig.from_dataset(
            config.dataset,
            config.exp_name
        )

    def run(self, syn_styles: List[Tuple], syn_texts: List[str], mel_config):
        """Execute random evaluation pipeline."""

        # Validate config
        self.config.validate()

        # Build pipeline
        pipeline = Pipeline([
            SynthesisStage(self.config.models),
            PSDExtractionStage(self.config.psd_level),
            StatisticsStage(),
            QualityEvaluationStage(),
        ])

        # Create context
        context = {
            "syn_styles": syn_styles,
            "syn_texts": syn_texts,
            "models": self.config.models,
            "output_dir": str(self.paths.output_dir),
            "style_syntex_name": self.config.style_syntex_name,
            "start_step": self.config.start_step,
            "end_step": self.config.end_step,
            "mel_config": mel_config,
            "save_attn": self.config.save_attn,
            "save_cond": self.config.save_cond,
            "save_attn_json": self.config.save_attn_json,
        }

        # Ensure output directory exists
        Path(self.paths.output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Random Evaluation: {self.config.name}")
        print(f"Dataset: {self.config.dataset}")
        print(f"Models: {', '.join(self.config.models)}")
        print(f"Output: {self.paths.output_dir}")
        print(f"Steps: {self.config.start_step} → {self.config.end_step}")
        print(f"{'='*60}\n")

        # Execute pipeline
        results = pipeline.run(context)

        print(f"\n{'='*60}")
        print("Random Evaluation Complete!")
        print(f"Results saved to: {self.paths.output_dir}")
        print(f"{'='*60}\n")

        return results
