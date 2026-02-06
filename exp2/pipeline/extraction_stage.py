"""PSD (Pitch/Energy/Duration) extraction pipeline stage."""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.append('/home/rosen/Project/StyleTTS2')

from exp2.pipeline.base_pipeline import PipelineStage
from exp.extract_psd import extract_psd, extract_psdave


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update into base."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _extract_model_psd_from_dict(prosody_dict: dict, model_name: str) -> dict:
    """Extract single model's PSD data from prosody_dict.

    Input format: {spk: {emo: {model_name: {pitch, energy, speechid}}}}
    Output format: {spk: {emo: {pitch, energy, speechid}}}  (without model_name level)
    """
    result = {}
    for spk in prosody_dict:
        result[spk] = {}
        for emo in prosody_dict[spk]:
            if model_name in prosody_dict[spk][emo]:
                result[spk][emo] = prosody_dict[spk][emo][model_name]
    return result


def _rebuild_prosody_dict_with_model(per_model_psd: dict, model_name: str) -> dict:
    """Rebuild prosody_dict format with model_name level.

    Input format: {spk: {emo: {pitch, energy, speechid}}}
    Output format: {spk: {emo: {model_name: {pitch, energy, speechid}}}}
    """
    result = {}
    for spk in per_model_psd:
        result[spk] = {}
        for emo in per_model_psd[spk]:
            result[spk][emo] = {model_name: per_model_psd[spk][emo]}
    return result


class PSDExtractionStage(PipelineStage):
    """Stage 1: Extract prosody features (pitch, energy, duration)."""

    def __init__(self, psd_level: str = "frame"):
        super().__init__(name="PSD Extraction", enabled=True)
        self.psd_level = psd_level

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract PSD features from synthesized speech.

        For each model:
        1. Extract PSD from speech_dir and save to model_dir/psd.json
        2. Aggregate all models into root psd_aggregated.json
        """
        models = context["models"]
        output_dir = Path(context["output_dir"])
        style_name = context.get("style_syntex_name", "random")
        mel_config = context["mel_config"]

        # Ensure reference is processed
        models_to_process = list(models)
        if "reference" not in models_to_process:
            models_to_process = ["reference"] + models_to_process

        print(f"Extracting {self.psd_level}-level PSD features...")

        # Process each model individually
        for model_name in models_to_process:
            model_dir = output_dir / model_name
            speech_dir = model_dir / style_name

            if not speech_dir.exists():
                print(f"⚠ Skipping {model_name}: directory {speech_dir} not found")
                continue

            print(f"  Processing {model_name}...")

            if self.psd_level == "frame":
                # Extract PSD for this single model
                model_prosody_dict = extract_psd(
                    mel_config,
                    str(speech_dir),
                    model_n=model_name,
                    save_psd_file="",
                    prosody_dict={}
                )

                # Extract just this model's data (without model_name level)
                per_model_psd = _extract_model_psd_from_dict(model_prosody_dict, model_name)

                # Save per-model PSD file inside speech_dir (supports fine-grained categories)
                # For random: model_dir/random/psd.json
                # For fine:   model_dir/05/psd.json, model_dir/1/psd.json, etc.
                per_model_psd_path = speech_dir / "psd.json"
                with open(per_model_psd_path, "w", encoding="utf-8") as f:
                    json.dump(per_model_psd, f, sort_keys=True, indent=4)
                print(f"    ✓ Saved {per_model_psd_path}")

            elif self.psd_level == "phoneme":
                # For phoneme-level, we need to pass all models at once
                # This is handled differently in the original code
                pass
            else:
                raise ValueError(f"Invalid psd_level: {self.psd_level}")

        # Aggregate all per-model PSDs into root file
        aggregated_psd = self._aggregate_psd_files(output_dir, models_to_process, style_name)

        # Save aggregated file (merge with existing)
        agg_path = output_dir / "psd_aggregated.json"
        if agg_path.exists():
            with open(agg_path, "r") as f:
                existing = json.load(f)
            aggregated_psd = _deep_merge(existing, aggregated_psd)

        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_psd, f, sort_keys=True, indent=4)
        print(f"✓ Aggregated PSD saved to {agg_path}")

        context["prosody_dict"] = aggregated_psd
        context["psd_json_path"] = str(agg_path)

        return context

    def _aggregate_psd_files(self, output_dir: Path, models: List[str], style_name: str) -> dict:
        """Load all per-model PSD files and combine into prosody_dict format."""
        aggregated = {}

        for model_name in models:
            # Per-model PSD is inside speech_dir (model_dir/style_name/psd.json)
            per_model_path = output_dir / model_name / style_name / "psd.json"

            if not per_model_path.exists():
                print(f"    ⚠ No PSD file for {model_name}, skipping aggregation")
                continue

            with open(per_model_path, "r") as f:
                per_model_psd = json.load(f)

            # Rebuild with model_name level and merge
            model_psd_with_key = _rebuild_prosody_dict_with_model(per_model_psd, model_name)
            aggregated = _deep_merge(aggregated, model_psd_with_key)

        return aggregated
