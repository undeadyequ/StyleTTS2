"""Speaker similarity evaluation pipeline stage (SIM-O, SIM-R)."""

import sys
import os
import json
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Any

sys.path.append('/home/rosen/Project/StyleTTS2')

from exp2.pipeline.base_pipeline import PipelineStage


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update into base."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class SpeakerSimilarityStage(PipelineStage):
    """Stage 4: Compute SIM-O and SIM-R speaker similarity scores.

    SIM-O: cosine similarity between synthesized speech and original reference.
    SIM-R: cosine similarity between synthesized speech and vocoded reference
           (domain-matched via HiFi-GAN vocoder).

    Uses WavLM-Large via Microsoft UniSpeech speaker verification tool.
    Requires s3prl to be installed: pip install s3prl
    """

    def __init__(
        self,
        vocoder_ckpt: str = '/home/rosen/ckpt/styletts/Vocoder/LibriTTS/',
        model_name: str = 'wavlm_large',
        need_syn_recon: bool = False,
    ):
        super().__init__(name="Speaker Similarity", enabled=True)
        self.vocoder_ckpt = vocoder_ckpt
        self.model_name = model_name
        self.need_syn_recon = need_syn_recon

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute SIM-O and SIM-R for all models.

        For each model:
        1. Pair synthesized wavs with reference wavs by stripping _synN suffix
        2. Compute SIM-O (vs original reference)
        3. Compute SIM-R (vs vocoded reference, cached per unique reference)
        4. Save per-pair scores to model_dir/style_name/sim_details.json
        5. Aggregate all models into root sim_aggregated.json
        """
        from exp.exec_speaker_sim import compute_sim_for_dir
        from load_vocoder import get_vocoder

        models = context["models"]
        output_dir = Path(context["output_dir"])
        style_name = context.get("style_syntex_name", "random")

        models_to_process = [m for m in models if m != "reference"]
        ref_dir = output_dir / "reference" / style_name
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if not ref_dir.exists():
            print(f"⚠ Reference directory not found: {ref_dir}, skipping SIM evaluation")
            context["sim_results"] = {}
            return context

        # Load vocoder and mel transform once for all models
        print(f"Loading vocoder from {self.vocoder_ckpt}...")
        vocoder = get_vocoder(self.vocoder_ckpt, device)
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

        print(f"\nComputing speaker similarity (SIM-O / SIM-R) with {self.model_name}...")

        all_sim = {}

        for model_name in models_to_process:
            model_dir = output_dir / model_name
            syn_dir = model_dir / style_name

            if not syn_dir.exists():
                print(f"⚠ Skipping {model_name}: directory {syn_dir} not found")
                continue

            print(f"\n  [{model_name}]")
            sim_json = syn_dir / "sim_details.json"

            result = compute_sim_for_dir(
                syn_dir=syn_dir,
                ref_dir=ref_dir,
                vocoder=vocoder,
                to_mel=to_mel,
                output_json=sim_json,
                model_name=self.model_name,
                device=device,
                need_syn_recon=self.need_syn_recon,
            )

            summary = {
                "sim_o_mean": result["sim_o_mean"],
                "sim_o_std":  result["sim_o_std"],
                "sim_r_mean": result["sim_r_mean"],
                "sim_r_std":  result["sim_r_std"],
            }
            all_sim[model_name] = summary

            print(f"    ✓ Saved {sim_json}")
            print(f"    SIM-O: {result['sim_o_mean']:.4f} ± {result['sim_o_std']:.4f}")
            print(f"    SIM-R: {result['sim_r_mean']:.4f} ± {result['sim_r_std']:.4f}")

        # Save aggregated (merge with existing)
        agg_path = output_dir / "sim_aggregated.json"
        if agg_path.exists():
            with open(agg_path, "r") as f:
                existing = json.load(f)
            merged = _deep_merge(existing, all_sim)
        else:
            merged = all_sim

        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, sort_keys=True, indent=4)

        print(f"\n✓ Aggregated speaker similarity saved to {agg_path}")

        context["sim_results"] = all_sim
        return context
