"""Statistics computation pipeline stage."""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

sys.path.append('/home/rosen/Project/StyleTTS2')

from exp2.pipeline.base_pipeline import PipelineStage
from exp.statcz_psd import statcz_psd_mcd
from exp.exp_utils import convert_json_to_pd2, convert_json_to_pd2_fine


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update into base."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _wrap_with_fine_name(data: dict, fine_name: str) -> dict:
    """Wrap statistics values with fine_name level for fine experiments.

    Input:  {spk: {emo: {model: values}}}
    Output: {spk: {emo: {model: {fine_name: values}}}}
    """
    result = {}
    for spk in data:
        result[spk] = {}
        for emo in data[spk]:
            result[spk][emo] = {}
            for model in data[spk][emo]:
                result[spk][emo][model] = {fine_name: data[spk][emo][model]}
    return result


def _wrap_with_fine_name_emo_mean(data: dict, fine_name: str) -> dict:
    """Wrap emo_mean statistics with fine_name level for fine experiments.

    Input:  {emo: {model: values}}
    Output: {emo: {model: {fine_name: values}}}
    """
    result = {}
    for emo in data:
        result[emo] = {}
        for model in data[emo]:
            result[emo][model] = {fine_name: data[emo][model]}
    return result


def _detect_fine_structure(data: dict) -> bool:
    """Detect if data has fine_name level (for fine experiments).

    Checks if the innermost level before values is a dict with string keys
    that map to lists (values), indicating fine_name structure.

    For spk_emo: {spk: {emo: {model: {fine_name: [values]}}}} -> True
    For random:  {spk: {emo: {model: [values]}}} -> False
    """
    try:
        for spk in data:
            for emo in data[spk]:
                for model in data[spk][emo]:
                    model_data = data[spk][emo][model]
                    # If model_data is a list, it's random structure
                    if isinstance(model_data, list):
                        return False
                    # If model_data is a dict, it's fine structure
                    if isinstance(model_data, dict):
                        return True
                    break
                break
            break
    except (KeyError, TypeError):
        pass
    return False


def _detect_fine_structure_emo_mean(data: dict) -> bool:
    """Detect if emo_mean data has fine_name level (for fine experiments).

    For emo_mean fine: {emo: {model: {fine_name: [values]}}} -> True
    For emo_mean random: {emo: {model: [values]}} -> False
    """
    try:
        for emo in data:
            for model in data[emo]:
                model_data = data[emo][model]
                # If model_data is a list, it's random structure
                if isinstance(model_data, list):
                    return False
                # If model_data is a dict, it's fine structure
                if isinstance(model_data, dict):
                    return True
                break
            break
    except (KeyError, TypeError):
        pass
    return False


def _build_mini_prosody_dict(ref_psd: dict, model_psd: dict, model_name: str) -> dict:
    """Build a prosody_dict with just reference + one model.

    Args:
        ref_psd: Reference PSD in format {spk: {emo: {pitch, energy, speechid}}}
        model_psd: Model PSD in same format
        model_name: Name of the model

    Returns:
        prosody_dict in format {spk: {emo: {"reference": {...}, model_name: {...}}}}
    """
    result = {}
    for spk in ref_psd:
        result[spk] = {}
        for emo in ref_psd[spk]:
            result[spk][emo] = {
                "reference": ref_psd[spk][emo]
            }
            # Add model data if available for this spk/emo
            if spk in model_psd and emo in model_psd[spk]:
                result[spk][emo][model_name] = model_psd[spk][emo]
    return result


def _spk_emo_to_multiindex_df(spk_emo_res: dict) -> pd.DataFrame:
    """Convert spk_emo stats to multiindex DataFrame with emotion columns.

    For random experiments:
        Input:  {spk: {emo: {model: [pitch_mean, energy_mean]}}}
        Output: DataFrame with MultiIndex (spk, model)

    For fine experiments:
        Input:  {spk: {emo: {model: {fine_name: [pitch_mean, energy_mean]}}}}
        Output: DataFrame with MultiIndex (spk, model, fine_name)
    """
    # Detect if this is fine experiment structure
    is_fine = _detect_fine_structure(spk_emo_res)

    # Collect all models, emotions, and fine_names
    all_models = set()
    all_emos = set()
    all_fine_names = set()

    for spk in spk_emo_res:
        for emo in spk_emo_res[spk]:
            all_emos.add(emo)
            for model in spk_emo_res[spk][emo]:
                all_models.add(model)
                if is_fine:
                    model_data = spk_emo_res[spk][emo][model]
                    if isinstance(model_data, dict):
                        all_fine_names.update(model_data.keys())

    rows = []

    if is_fine:
        # Fine experiment: 3-level MultiIndex (spk, model, fine_name)
        for spk in sorted(spk_emo_res.keys()):
            for model in sorted(all_models):
                for fine_name in sorted(all_fine_names):
                    row = {"spk": spk, "model": model, "fine_name": fine_name}
                    for emo in sorted(all_emos):
                        if (emo in spk_emo_res[spk] and
                            model in spk_emo_res[spk][emo] and
                            isinstance(spk_emo_res[spk][emo][model], dict) and
                            fine_name in spk_emo_res[spk][emo][model]):
                            values = spk_emo_res[spk][emo][model][fine_name]
                            if isinstance(values, list) and len(values) >= 2:
                                row[f"{emo}_pitch"] = values[0]
                                row[f"{emo}_energy"] = values[1]
                    rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index(["spk", "model", "fine_name"])
    else:
        # Random experiment: 2-level MultiIndex (spk, model)
        for spk in sorted(spk_emo_res.keys()):
            for model in sorted(all_models):
                row = {"spk": spk, "model": model}
                for emo in sorted(all_emos):
                    if emo in spk_emo_res[spk] and model in spk_emo_res[spk][emo]:
                        values = spk_emo_res[spk][emo][model]
                        if isinstance(values, list) and len(values) >= 2:
                            row[f"{emo}_pitch"] = values[0]
                            row[f"{emo}_energy"] = values[1]
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index(["spk", "model"])

    # Sort columns: group by emotion (emo_pitch, emo_energy)
    sorted_cols = []
    for emo in sorted(all_emos):
        if f"{emo}_pitch" in df.columns:
            sorted_cols.append(f"{emo}_pitch")
        if f"{emo}_energy" in df.columns:
            sorted_cols.append(f"{emo}_energy")
    df = df[sorted_cols]

    return df


class StatisticsStage(PipelineStage):
    """Stage 2: Compute statistics (DTW) from PSD features."""

    def __init__(self):
        super().__init__(name="Statistics Computation", enabled=True)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute DTW statistics from prosody dict.

        For each model:
        1. Load reference PSD and model PSD
        2. Compute stats (DTW)
        3. Save to model_dir/stats.json
        4. Aggregate all models into root stats_aggregated files
        """
        output_dir = Path(context["output_dir"])
        style_name = context.get("style_syntex_name", "random")
        models = context["models"]

        # Don't include reference in models to process
        models_to_process = [m for m in models if m != "reference"]

        print("Computing DTW statistics...")

        # Load reference PSD once (inside speech_dir: reference/style_name/psd.json)
        ref_psd_path = output_dir / "reference" / style_name / "psd.json"
        if not ref_psd_path.exists():
            # Fallback to old location (directly under reference/)
            ref_psd_path = output_dir / "reference" / "psd.json"
        if not ref_psd_path.exists():
            print(f"⚠ Reference PSD not found, skipping statistics")
            return context

        with open(ref_psd_path, "r") as f:
            ref_psd = json.load(f)

        all_dtw_res = {}
        all_spk_emo_res = {}  # Per-speaker-emotion stats
        all_emo_mean_res = {}  # Mean over speakers (per emotion)

        # Process each model individually
        for model_name in models_to_process:
            model_dir = output_dir / model_name
            speech_dir = model_dir / style_name
            # Try speech_dir first (supports fine-grained categories), then fallback
            model_psd_path = speech_dir / "psd.json"
            if not model_psd_path.exists():
                # Fallback to old location (directly under model_dir)
                model_psd_path = model_dir / "psd.json"

            if not model_psd_path.exists():
                print(f"  ⚠ Skipping {model_name}: PSD file not found")
                continue

            print(f"  Processing {model_name}...")

            # Load model's PSD
            with open(model_psd_path, "r") as f:
                model_psd = json.load(f)

            # Build mini prosody_dict with just reference + this model
            mini_dict = _build_mini_prosody_dict(ref_psd, model_psd, model_name)

            # Compute stats for this model
            # dtw_res: per-utterance DTW scores {spk: {emo: {model: [[pitch], [energy], [speechid]]}}}
            # spk_emo_res: per-speaker-emotion stats {spk: {emo: {model: [pitch_mean, energy_mean]}}}
            # emo_mean_res: mean over speakers {emo: {model: [pitch_mean, energy_mean]}}
            dtw_res, spk_emo_res, emo_mean_res = statcz_psd_mcd(mini_dict, exclude_zero=True)

            # Save per-model stats inside speech_dir (supports fine-grained categories)
            per_model_stats = {
                "dtw": dtw_res,
                "spk_emo": spk_emo_res,
                "emo_mean": emo_mean_res
            }
            stats_path = speech_dir / "stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(per_model_stats, f, sort_keys=True, indent=4)
            print(f"    ✓ Saved {stats_path}")

            # Detect if this is a fine experiment (style_name != "random")
            is_fine_experiment = (style_name != "random")

            # Collect for aggregation - wrap with fine_name for fine experiments
            if is_fine_experiment:
                # Wrap with fine_name level: {spk: {emo: {model: {fine_name: values}}}}
                wrapped_dtw = _wrap_with_fine_name(dtw_res, style_name)
                wrapped_spk_emo = _wrap_with_fine_name(spk_emo_res, style_name)
                wrapped_emo_mean = _wrap_with_fine_name_emo_mean(emo_mean_res, style_name)
            else:
                # Random experiment: use existing structure
                wrapped_dtw = dtw_res
                wrapped_spk_emo = spk_emo_res
                wrapped_emo_mean = emo_mean_res

            all_dtw_res = _deep_merge(all_dtw_res, wrapped_dtw)
            all_spk_emo_res = _deep_merge(all_spk_emo_res, wrapped_spk_emo)
            all_emo_mean_res = _deep_merge(all_emo_mean_res, wrapped_emo_mean)

        # Save aggregated statistics (merge with existing)
        agg_dtw_path = output_dir / "dtw_aggregated.json"

        # Merge DTW with existing if present
        if agg_dtw_path.exists():
            with open(agg_dtw_path, "r") as f:
                existing_dtw = json.load(f)
            merged_dtw = _deep_merge(existing_dtw, all_dtw_res)
        else:
            merged_dtw = all_dtw_res

        with open(agg_dtw_path, "w", encoding="utf-8") as f:
            json.dump(merged_dtw, f, sort_keys=True, indent=4)

        # Create helper files directory and save intermediate JSON files there
        helper_dir = output_dir / "aggregation_help_files"
        helper_dir.mkdir(exist_ok=True)

        # Save spk_emo JSON to helper directory (merge with existing)
        agg_spk_emo_json_path = helper_dir / "stats_spk_emo_aggregated.json"
        if agg_spk_emo_json_path.exists():
            with open(agg_spk_emo_json_path, "r") as f:
                existing_spk_emo_json = json.load(f)
            merged_spk_emo_json = _deep_merge(existing_spk_emo_json, all_spk_emo_res)
        else:
            merged_spk_emo_json = all_spk_emo_res

        with open(agg_spk_emo_json_path, "w", encoding="utf-8") as f:
            json.dump(merged_spk_emo_json, f, sort_keys=True, indent=4)

        # Use the merged JSON for CSV generation (JSON is source of truth)
        agg_spk_emo_csv_path = output_dir / "stats_spk_emo_mean_aggregated.csv"
        merged_spk_emo = merged_spk_emo_json  # Use the already-merged JSON data

        # Also merge emo_mean for the emo_mean CSV
        agg_emo_mean_csv_path = output_dir / "stats_emo_mean_aggregated.csv"
        agg_emo_mean_json_path = helper_dir / "stats_emo_mean_aggregated.json"
        if agg_emo_mean_json_path.exists():
            with open(agg_emo_mean_json_path, "r") as f:
                existing_emo_mean = json.load(f)
            merged_emo_mean = _deep_merge(existing_emo_mean, all_emo_mean_res)
        else:
            merged_emo_mean = all_emo_mean_res

        # Save emo_mean JSON to helper directory
        with open(agg_emo_mean_json_path, "w", encoding="utf-8") as f:
            json.dump(merged_emo_mean, f, sort_keys=True, indent=4)

        print(f"✓ Aggregated statistics saved to:")
        print(f"  - DTW (per-utterance): {agg_dtw_path}")
        print(f"  - Helper files: {helper_dir}/")

        # Convert to pandas and save CSVs (using merged data)
        try:
            # 1. spk × model multiindex CSV (with emotion columns)
            if merged_spk_emo:
                spk_emo_df = _spk_emo_to_multiindex_df(merged_spk_emo)
                spk_emo_df.to_csv(agg_spk_emo_csv_path, index=True)
                print(f"  - CSV (spk x model multiIndex): {agg_spk_emo_csv_path}")

            # 2. emo × model multiindex CSV and model mean CSV
            if merged_emo_mean:
                # Detect if emo_mean has fine structure: {emo: {model: {fine_name: values}}}
                is_emo_mean_fine = _detect_fine_structure_emo_mean(merged_emo_mean)

                if is_emo_mean_fine:
                    # Fine experiment: use convert_json_to_pd2_fine
                    pivot_df_pitch, pivot_df_energy = convert_json_to_pd2_fine(
                        merged_emo_mean,
                        need_multi_index=True,
                        need_print_latex=True
                    )

                    # Save separate pitch and energy CSVs
                    pitch_csv_path = output_dir / "stats_emo_mean_pitch_aggregated.csv"
                    energy_csv_path = output_dir / "stats_emo_mean_energy_aggregated.csv"
                    pivot_df_pitch.to_csv(pitch_csv_path, index=True)
                    pivot_df_energy.to_csv(energy_csv_path, index=True)

                    print(f"  - CSV (emo_mean pitch, fine): {pitch_csv_path}")
                    print(f"  - CSV (emo_mean energy, fine): {energy_csv_path}")

                    # Print statistics table
                    print("\nStatistics Summary (pitch, fine experiment):")
                    print(pivot_df_pitch)
                    print("\nStatistics Summary (energy, fine experiment):")
                    print(pivot_df_energy)
                else:
                    # Random experiment: use convert_json_to_pd2
                    emo_mean_multiindex_df, model_mean_df = convert_json_to_pd2(
                        merged_emo_mean,
                        need_multi_index=True,
                        need_print_latex=True
                    )

                    # stats_emo_mean_aggregated.csv: multiIndex (emotion x model)
                    emo_mean_multiindex_df.to_csv(agg_emo_mean_csv_path, index=True)

                    # stats_mean_aggregated.csv: mean over emotions for each model
                    mean_csv_path = output_dir / "stats_mean_aggregated.csv"
                    model_mean_df.to_csv(mean_csv_path, index=True)

                    print(f"  - CSV (emo x model multiIndex): {agg_emo_mean_csv_path}")
                    print(f"  - CSV (model mean): {mean_csv_path}")

                    # Print statistics table
                    print("\nStatistics Summary (per emotion):")
                    print(emo_mean_multiindex_df)
                    print("\nStatistics Summary (model mean):")
                    print(model_mean_df)

        except Exception as e:
            print(f"  ⚠ Could not generate CSV summary: {e}")
            import traceback
            traceback.print_exc()

        context["psd_dtw_res"] = merged_dtw
        context["psd_spk_emo_stats"] = merged_spk_emo
        context["psd_emo_mean_stats"] = merged_emo_mean

        return context

    def _df_to_spk_emo_dict(self, df: pd.DataFrame) -> dict:
        """Convert MultiIndex (spk × model) DataFrame back to spk_emo_res format.

        Input: DataFrame with multiindex (spk, model), columns: emo_pitch, emo_energy
        Output: {spk: {emo: {model: [pitch_mean, energy_mean]}}}
        """
        result = {}

        # Extract emotion names from columns (e.g., 'angry_pitch' -> 'angry')
        emotions = set()
        for col in df.columns:
            if col.endswith('_pitch'):
                emotions.add(col[:-6])  # Remove '_pitch' suffix

        for (spk, model), row in df.iterrows():
            if spk not in result:
                result[spk] = {}
            for emo in emotions:
                pitch_col = f"{emo}_pitch"
                energy_col = f"{emo}_energy"
                if pitch_col in row and energy_col in row:
                    pitch_val = row[pitch_col]
                    energy_val = row[energy_col]
                    # Skip NaN values
                    if pd.notna(pitch_val) and pd.notna(energy_val):
                        if emo not in result[spk]:
                            result[spk][emo] = {}
                        # Convert to float in case they're strings from CSV
                        result[spk][emo][model] = [float(pitch_val), float(energy_val)]

        return result

    def _df_to_emo_mean_dict(self, df: pd.DataFrame) -> dict:
        """Convert MultiIndex (emo × model) DataFrame back to emo_mean_res format.

        Input: DataFrame with multiindex (emo, model), columns: pitch, energy (or similar)
        Output: {emo: {model: [pitch_mean, energy_mean]}}
        """
        result = {}

        for (emo, model), row in df.iterrows():
            if emo not in result:
                result[emo] = {}
            # Assume columns are pitch and energy (or first two numeric columns)
            values = row.values.tolist()
            if len(values) >= 2:
                # Convert to float in case they're strings from CSV
                result[emo][model] = [float(values[0]), float(values[1])]

        return result
