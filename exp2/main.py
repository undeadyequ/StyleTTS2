#!/usr/bin/env python3
"""
Refactored experimental evaluation script for StyleTTS2.

Usage:
    python exp2/main.py random --models monoDiT styletts2 --dataset esd
    python exp2/main.py grid --base-model decodit_cfm_v29 --epochs 48 50 --dataset esd
    python exp2/main.py fine --category length_ratio --models monoDiT --dataset esd
"""

import sys
import os
import argparse
import torch

# Add project root to path
sys.path.append('/home/rosen/Project/StyleTTS2')
os.chdir('/home/rosen/Project/StyleTTS2')

from exp2.config.base_config import ExperimentConfig
from exp2.config.ablation_config import AblationGrid
from exp2.experiments.random_eval import RandomEvaluation
from exp2.experiments.grid_ablation import GridAblationStudy
from exp2.experiments.fine_grained_eval import FineGrainedEvaluation, FineGrainedConfig
from exp.exp_utils import get_synStyle_from_file, get_synText_from_file
from mel_config import MelConfig

def run_random_evaluation(args):
    """Run standard random evaluation."""

    print(f"\n{'='*70}")
    print("RANDOM SYNTHESIS EVALUATION")
    print(f"{'='*70}\n")

    # Create experiment config
    config = ExperimentConfig(
        name="random_eval",
        dataset=args.dataset,
        models=args.models,
        start_step=args.start_step,
        end_step=args.end_step,
        save_attn=args.save_attn,
        save_attn_json=args.save_attn_json,
        save_cond=args.save_cond,
        psd_level=args.psd_level,
        style_syntex_name="random",
        exp_name = args.exp_name
    )

    # Load data
    syn_styles = get_synStyle_from_file(
        args.style_file,
        split_char='|',
        melstyle_type="codec",
        dataset_name=args.dataset
    )
    syn_texts = get_synText_from_file(args.text_file)

    print(f"Loaded {len(syn_styles)} styles and {len(syn_texts)} texts")

    # Run evaluation
    evaluator = RandomEvaluation(config)
    results = evaluator.run(syn_styles, syn_texts, MelConfig)

    print(f"\n✓ Evaluation complete!")
    return results


def run_fine_grained_eval(args):
    """Run fine-grained analysis.

    Fine-grained evaluation uses multiple text files (one per fine category)
    with a shared style file. For example:
    - length_ratio: s2_05.txt, s2_1.txt, s2_2.txt with r1_50.txt
    - position: s3_same_pos.txt, s3_diff_pos.txt with r2.txt
    """

    print(f"\n{'='*70}")
    print(f"FINE-GRAINED EVALUATION: {args.category.upper()}")
    print(f"{'='*70}\n")

    # Get fine-grained config
    if args.category in ["length_ratio", "position"]:
        fine_config = FineGrainedConfig.get_fine_config(args.category)
        fine_labels = fine_config["labels"]
        text_pattern = fine_config["text_pattern"]                               # decided by fine_config
        style_file = args.style_file or f"exp/data2/{fine_config['style_file']}"  # decide by fine_config or args input
        print(f"Using predefined config: {fine_config['description']}")
    else:
        # Custom fine categories from CLI
        fine_labels = args.fine_labels
        text_pattern = args.text_pattern
        style_file = args.style_file

    # Build fine_categories dict: {label: [texts]}
    fine_categories = {}
    text_dir = args.text_dir or "exp/data2"
    for label in fine_labels:
        text_file = os.path.join(text_dir, text_pattern.format(label=label))
        if os.path.exists(text_file):
            texts = get_synText_from_file(text_file)
            fine_categories[label] = texts
            print(f"  Category '{label}': {len(texts)} texts from {text_file}")
        else:
            print(f"  ⚠ Warning: Text file not found: {text_file}")

    if not fine_categories:
        print("Error: No valid fine categories found!")
        return None

    # Load styles (shared across all fine categories)
    print(style_file)
    syn_styles = get_synStyle_from_file(
        style_file,
        split_char='|',
        melstyle_type="codec",
        dataset_name=args.dataset
    )
    print(f"\nLoaded {len(syn_styles)} styles from {style_file}")

    # Create experiment config
    config = ExperimentConfig(
        name=f"fine_{args.category}",
        dataset=args.dataset,
        models=args.models,
        start_step=args.start_step,
        end_step=args.end_step,
        save_attn=args.save_attn,
        save_attn_json=args.save_attn_json,
        save_cond=args.save_cond,
        psd_level=args.psd_level,
        style_syntex_name="fine",  # Will be overridden per category
        exp_name=args.exp_name
    )

    # Run fine-grained evaluation
    evaluator = FineGrainedEvaluation(config)
    results = evaluator.run(syn_styles, fine_categories, MelConfig)

    print(f"\n✓ Fine-grained evaluation complete!")
    return results


def run_grid_ablation(args):
    """Run grid-based ablation study."""

    print(f"\n{'='*70}")
    print("GRID ABLATION STUDY")
    print(f"{'='*70}\n")

    # Build ablation grid from CLI args
    param_grid = {}
    if args.trend_strengths:
        param_grid["trend_strength"] = args.trend_strengths
    if args.cfg_strengths:
        param_grid["cfg_strength"] = args.cfg_strengths
    if args.drop_trends:
        param_grid["drop_trend"] = args.drop_trends

    ablation_grid = AblationGrid(
        base_model=args.base_model,
        epochs=args.epochs if args.epochs else [],
        param_grid=param_grid
    )

    # Create experiment config
    config = ExperimentConfig(
        name=f"grid_ablation_{args.base_model}",
        dataset=args.dataset,
        models=[],  # Will be populated by variants
        start_step=args.start_step,
        end_step=args.end_step,
        save_attn=args.save_attn,
        save_attn_json=args.save_attn_json,
        save_cond=args.save_cond,
        psd_level=args.psd_level,
        style_syntex_name="random",
        exp_name=args.exp_name
    )

    # Load data
    syn_styles = get_synStyle_from_file(
        args.style_file,
        split_char='|',
        melstyle_type="codec",
        dataset_name=args.dataset
    )
    syn_texts = get_synText_from_file(args.text_file)

    print(f"Loaded {len(syn_styles)} styles and {len(syn_texts)} texts")

    # Run grid ablation study
    study = GridAblationStudy(config, ablation_grid)
    results = study.run(syn_styles, syn_texts, MelConfig)

    print(f"\n Grid ablation study complete!")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="TTS Model Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
        Examples:
        # Random evaluation with multiple models
        python exp2/main.py random --models monoDiT styletts2 DiT --dataset esd

        # Grid ablation with different epochs and parameters
        python exp2/main.py grid --base-model decodit_cfm_v29 --epochs 48 50 --trend-strengths 3.0 5.0 --dataset esd

        # Fine-grained length ratio analysis
        python exp2/main.py fine --category length_ratio --models monoDiT --dataset esd
        """)

    subparsers = parser.add_subparsers(dest="command", help="Evaluation mode")

    # ===== Random evaluation =====
    random_parser = subparsers.add_parser("random", help="Random synthesis evaluation")
    random_parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate", default=["decodit_cfm_v24"])
    random_parser.add_argument("--dataset", choices=["esd", "libritts"], required=True, help="Dataset name", default="esd")
    random_parser.add_argument("--style-file", default="exp/data/r1_50.txt", help="Style file path")
    random_parser.add_argument("--text-file", default="exp/data/s1_5.txt", help="Text file path")
    random_parser.add_argument("--start-step", type=int, default=0, help="Start step (0-6)")
    random_parser.add_argument("--end-step", type=int, default=0, help="End step (0-6)")
    random_parser.add_argument("--psd-level", choices=["frame", "phoneme"], default="frame", help="PSD extraction level")
    random_parser.add_argument("--save-attn", action="store_true", help="Save attention maps", default=False)
    random_parser.add_argument("--save-attn-json", action="store_true", help="Save attention JSON", default=False)
    random_parser.add_argument("--save-cond", action="store_true", help="Save conditioning info", default=False)
    random_parser.add_argument("--exp-name", default="mdit_tts", help="prefix of experiment output dir")

    # ===== Fine-grained evaluation =====
    fine_parser = subparsers.add_parser("fine", help="Fine-grained analysis")
    fine_parser.add_argument("--category", required=True,
                             help="Analysis category: 'length_ratio', 'position', or custom name")
    fine_parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate")
    fine_parser.add_argument("--dataset", choices=["esd", "libritts"], required=True, help="Dataset name")
    fine_parser.add_argument("--style-file", default=None,
                             help="Style file path (default: use predefined for category)")
    fine_parser.add_argument("--text-dir", default="exp/data2",
                             help="Directory containing text files")
    fine_parser.add_argument("--text-pattern", default="fine_esd_syn_ratio{label}.txt",
                             help="Text file pattern with {label} placeholder")
    fine_parser.add_argument("--fine-labels", nargs="+", default=None,
                             help="Fine category labels (e.g., 05 1 2)")
    fine_parser.add_argument("--start-step", type=int, default=0, help="Start step (0-3)")
    fine_parser.add_argument("--end-step", type=int, default=3, help="End step (0-3)")
    fine_parser.add_argument("--psd-level", choices=["frame", "phoneme"], default="frame",
                             help="PSD extraction level")
    fine_parser.add_argument("--save-attn", action="store_true", help="Save attention maps")
    fine_parser.add_argument("--save-attn-json", action="store_true", help="Save attention JSON")
    fine_parser.add_argument("--save-cond", action="store_true", help="Save conditioning info")
    fine_parser.add_argument("--exp-name", default="fine_eval",
                             help="Prefix of experiment output dir")

    # ===== Grid ablation study =====
    grid_parser = subparsers.add_parser("grid", help="Grid-based ablation study")
    grid_parser.add_argument("--base-model", required=True,
                             help="Base model name (e.g., decodit_cfm_v29)")
    grid_parser.add_argument("--epochs", nargs="+", type=int, default=[],
                             help="Epochs to evaluate (e.g., 48 50)")
    grid_parser.add_argument("--trend-strengths", nargs="+", type=float, default=[],
                             help="Trend strength values (e.g., 3.0 5.0)")
    grid_parser.add_argument("--cfg-strengths", nargs="+", type=float, default=[],
                             help="CFG strength values")
    grid_parser.add_argument("--drop-trends", nargs="+", type=lambda x: x.lower() == 'true',
                             default=[], help="Drop trend values (true/false)")
    grid_parser.add_argument("--dataset", choices=["esd", "libritts"], required=True,
                             help="Dataset name")
    grid_parser.add_argument("--style-file", default="exp/data/r1_50.txt",
                             help="Style file path")
    grid_parser.add_argument("--text-file", default="exp/data/s1_5.txt",
                             help="Text file path")
    grid_parser.add_argument("--start-step", type=int, default=0, help="Start step (0-3)")
    grid_parser.add_argument("--end-step", type=int, default=3, help="End step (0-3)")
    grid_parser.add_argument("--psd-level", choices=["frame", "phoneme"], default="frame",
                             help="PSD extraction level")
    grid_parser.add_argument("--save-attn", action="store_true", help="Save attention maps")
    grid_parser.add_argument("--save-attn-json", action="store_true", help="Save attention JSON")
    grid_parser.add_argument("--save-cond", action="store_true", help="Save conditioning info")
    grid_parser.add_argument("--exp-name", default="grid_ablation",
                             help="Prefix of experiment output dir")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(0)

    # Dispatch to appropriate handler
    if args.command == "random":
        run_random_evaluation(args)
    elif args.command == "fine":
        run_fine_grained_eval(args)
    elif args.command == "grid":
        run_grid_ablation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
