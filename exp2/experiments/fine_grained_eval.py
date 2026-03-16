"""Fine-grained evaluation experiment.

Fine-grained experiments evaluate models across multiple style-text conditions,
where each condition (e.g., length_ratio: "05", "1", "2") has its own text file
but shares the same style/reference file.

Output structure:
    output_dir/
    ├── modelA/
    │   ├── 05/           # Fine category "05"
    │   │   ├── speech files...
    │   │   └── wer_details.csv, utmos_details.txt
    │   ├── 1/            # Fine category "1"
    │   └── 2/            # Fine category "2"
    ├── modelA/
    │   ├── psd.json      # Aggregated PSD across all fine categories
    │   ├── stats.json    # Aggregated stats
    │   └── quality.json
    ├── reference/
    │   ├── 05/, 1/, 2/   # Reference speech for each fine category
    │   └── psd.json
    └── aggregated files...
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Any, Dict

sys.path.append('/home/rosen/Project/StyleTTS2')

from exp2.config.base_config import ExperimentConfig, PathConfig
from exp2.pipeline.base_pipeline import Pipeline
from exp2.pipeline.synthesis_stage import SynthesisStage
from exp2.pipeline.extraction_stage import PSDExtractionStage
from exp2.pipeline.statistics_stage import StatisticsStage
from exp2.pipeline.quality_stage import QualityEvaluationStage
from exp2.pipeline.speaker_sim_stage import SpeakerSimilarityStage


class FineGrainedEvaluation:
    """Fine-grained evaluation across multiple style-text conditions.

    Unlike random evaluation which uses a single style_syntex_name,
    fine-grained evaluation iterates over multiple fine categories
    (e.g., length ratios "05", "1", "2") with different text files.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.paths = PathConfig.from_dataset(
            config.dataset,
            config.exp_name
        )

    def run(
        self,
        syn_styles: List[Tuple],
        fine_categories: Dict[str, List[str]],
        mel_config,
    ):
        """Execute fine-grained evaluation pipeline.

        Args:
            syn_styles: List of style tuples (shared across all fine categories)
            fine_categories: Dict mapping category label to list of texts
                e.g., {"05": ["text1", "text2"], "1": ["text1", "text2"], "2": [...]}
            mel_config: Mel spectrogram configuration
        """

        # Validate config
        self.config.validate()

        # Ensure output directory exists
        Path(self.paths.output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Fine-Grained Evaluation: {self.config.name}")
        print(f"Dataset: {self.config.dataset}")
        print(f"Models: {', '.join(self.config.models)}")
        print(f"Fine categories: {list(fine_categories.keys())}")
        print(f"Output: {self.paths.output_dir}")
        print(f"Steps: {self.config.start_step} → {self.config.end_step}")
        print(f"{'='*60}\n")

        all_results = {}

        # Phase 1: Synthesis for each fine category
        if self.config.start_step <= 0 <= self.config.end_step:
            print("\n" + "="*50)
            print("PHASE 1: SYNTHESIS")
            print("="*50)

            for fine_label, syn_texts in fine_categories.items():
                print(f"\n--- Synthesizing for fine category: {fine_label} ---")

                # Build synthesis-only pipeline
                synthesis_pipeline = Pipeline([
                    SynthesisStage(self.config.models),
                ])

                context = {
                    "syn_styles": syn_styles,
                    "syn_texts": syn_texts,
                    "models": self.config.models,
                    "output_dir": str(self.paths.output_dir),
                    "style_syntex_name": fine_label,
                    "start_step": 0,
                    "end_step": 0,
                    "mel_config": mel_config,
                    "save_attn": self.config.save_attn,
                    "save_cond": self.config.save_cond,
                    "save_attn_json": self.config.save_attn_json,
                }

                synthesis_pipeline.run(context)

        # Phase 2: PSD Extraction for each fine category
        if self.config.start_step <= 1 <= self.config.end_step:
            print("\n" + "="*50)
            print("PHASE 2: PSD EXTRACTION")
            print("="*50)

            for fine_label in fine_categories.keys():
                print(f"\n--- Extracting PSD for fine category: {fine_label} ---")

                extraction_pipeline = Pipeline([
                    PSDExtractionStage(self.config.psd_level),
                ])
                context = {
                    "models": self.config.models,
                    "output_dir": str(self.paths.output_dir),
                    "style_syntex_name": fine_label,
                    "start_step": 0,  # Stage index within this pipeline
                    "end_step": 0,
                    "mel_config": mel_config,
                }
                extraction_pipeline.run(context)

        # Phase 3: Statistics computation (aggregated across fine categories)
        if self.config.start_step <= 2 <= self.config.end_step:
            print("\n" + "="*50)
            print("PHASE 3: STATISTICS COMPUTATION")
            print("="*50)

            # Derive category name from config name (e.g. "fine_position" → "position")
            fine_category_name = self.config.name.removeprefix("fine_")

            # Process each fine category
            for fine_label in fine_categories.keys():
                print(f"\n--- Computing statistics for fine category: {fine_label} ---")

                statistics_pipeline = Pipeline([
                    StatisticsStage(),
                ])

                context = {
                    "models": self.config.models,
                    "output_dir": str(self.paths.output_dir),
                    "style_syntex_name": fine_label,
                    "fine_category_name": fine_category_name,
                    "start_step": 0,  # Stage index within this pipeline
                    "end_step": 0,
                    "mel_config": mel_config,
                }

                statistics_pipeline.run(context)

        # Phase 4: Quality evaluation for each fine category
        if self.config.start_step <= 3 <= self.config.end_step:
            print("\n" + "="*50)
            print("PHASE 4: QUALITY EVALUATION")
            print("="*50)

            for fine_label in fine_categories.keys():
                print(f"\n--- Evaluating quality for fine category: {fine_label} ---")

                quality_pipeline = Pipeline([
                    QualityEvaluationStage(),
                ])

                context = {
                    "models": self.config.models,
                    "output_dir": str(self.paths.output_dir),
                    "style_syntex_name": fine_label,
                    "start_step": 0,  # Stage index within this pipeline
                    "end_step": 0,
                    "mel_config": mel_config,
                }

                results = quality_pipeline.run(context)
                all_results[fine_label] = results.get("wer_utmos_results", {})

        # Phase 5: Speaker similarity for each fine category
        if self.config.start_step <= 4 <= self.config.end_step:
            print("\n" + "="*50)
            print("PHASE 5: SPEAKER SIMILARITY (SIM-O / SIM-R)")
            print("="*50)

            for fine_label in fine_categories.keys():
                print(f"\n--- Computing speaker similarity for fine category: {fine_label} ---")

                sim_pipeline = Pipeline([
                    SpeakerSimilarityStage(),
                ])

                context = {
                    "models": self.config.models,
                    "output_dir": str(self.paths.output_dir),
                    "style_syntex_name": fine_label,
                    "start_step": 0,
                    "end_step": 0,
                    "mel_config": mel_config,
                }

                results = sim_pipeline.run(context)
                all_results.setdefault(fine_label, {})
                all_results[fine_label]["sim_results"] = results.get("sim_results", {})

        print(f"\n{'='*60}")
        print("Fine-Grained Evaluation Complete!")
        print(f"Results saved to: {self.paths.output_dir}")
        print(f"{'='*60}\n")

        return all_results


class FineGrainedConfig:
    """Configuration for fine-grained experiments."""

    # Predefined fine categories
    LENGTH_RATIO = {
        "name": "length_ratio",
        "labels": ["05", "1", "2"],
        "text_pattern": "fine_esd_syn_ratio{label}.txt",
        "style_file": "fine_esd_ref.txt",
        "description": "Evaluation with different syn/ref length ratios (0.5x, 1x, 2x)"
    }

    POSITION = {
        "name": "position",
        "labels": ["same_pos", "diff_pos"],
        "text_pattern": "fine_esd_syn_{label}.txt",
        "style_file": "fine_esd_ref_pos.txt",
        "description": "Evaluation with same/different emphasis positions"
    }

    @classmethod
    def get_fine_config(cls, category: str) -> dict:
        """Get predefined fine-grained config by category name."""
        configs = {
            "length_ratio": cls.LENGTH_RATIO,
            "position": cls.POSITION,
        }
        if category not in configs:
            raise ValueError(f"Unknown fine category: {category}. "
                           f"Available: {list(configs.keys())}")
        return configs[category]
