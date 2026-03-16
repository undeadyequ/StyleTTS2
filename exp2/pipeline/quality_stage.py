"""Quality evaluation pipeline stage (WER, UTMOS)."""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

sys.path.append('/home/rosen/Project/StyleTTS2')

from exp2.pipeline.base_pipeline import PipelineStage
from exp.exp_wer import evaluate_wer
from exp.exec_utmosv2 import run_utmos


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update into base."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class QualityEvaluationStage(PipelineStage):
    """Stage 3: Evaluate speech quality (WER, UTMOS-v2)."""

    def __init__(self):
        super().__init__(name="Quality Evaluation", enabled=True)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate WER and UTMOS for all models.

        For each model:
        1. Compute WER and UTMOS from speech in model_dir/style_name/
        2. Save to model_dir/quality.json
        3. Aggregate all models into root quality_aggregated.json
        """
        models = context["models"]
        output_dir = Path(context["output_dir"])
        style_name = context.get("style_syntex_name", "random")

        # Don't include reference in models to evaluate
        models_to_process = [m for m in models if m != "reference"]

        all_quality = {}

        print("Evaluating speech quality (WER, UTMOS-v2)...")

        for model_name in models_to_process:
            model_dir = output_dir / model_name
            speech_dir = model_dir / style_name

            if not speech_dir.exists():
                print(f"⚠ Skipping {model_name}: directory not found")
                continue

            print(f"\n  Evaluating {model_name}...")

            # WER evaluation (details saved in speech_dir)
            wer_csv_path = speech_dir / "wer_details.csv"
            wer, sub, dele, ins = evaluate_wer(
                str(speech_dir),
                output_csv=str(wer_csv_path)
            )

            # UTMOS evaluation (details saved in speech_dir)
            utmos_txt_path = speech_dir / "utmos_details.txt"
            mean_mos, std_mos, mos_list = run_utmos(
                str(speech_dir),
                output_file=str(utmos_txt_path)
            )

            # Per-model quality metrics
            quality = {
                "wer": 0, #wer
                "substitutions": 0, #sub
                "deletions": 0, # dele
                "insertions": 0, # ins
                "utmos_mean": mean_mos,
                "utmos_std": std_mos
            }
            quality = {
                "wer": wer, #wer
                "substitutions": sub, #sub
                "deletions": dele, # dele
                "insertions": ins, # ins
                "utmos_mean": mean_mos,
                "utmos_std": std_mos
            }

            # Save per-model quality file inside speech_dir (supports fine-grained categories)
            quality_path = speech_dir / "quality.json"
            with open(quality_path, "w", encoding="utf-8") as f:
                json.dump(quality, f, sort_keys=True, indent=4)
            print(f"    ✓ Saved {quality_path}")

            print(f"    WER: {wer:.4f} (sub={sub:.4f}, del={dele:.4f}, ins={ins:.4f})")
            print(f"    UTMOS-v2: {mean_mos:.4f} ± {std_mos:.4f}")

            # Collect for aggregation
            all_quality[model_name] = quality

        # Save aggregated quality metrics (merge with existing)
        agg_path = output_dir / "quality_aggregated.json"

        if agg_path.exists():
            with open(agg_path, "r") as f:
                existing = json.load(f)
            merged = _deep_merge(existing, all_quality)
        else:
            merged = all_quality

        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, sort_keys=True, indent=4)

        print(f"\n✓ Aggregated quality metrics saved to {agg_path}")

        context["wer_utmos_results"] = all_quality

        return context
