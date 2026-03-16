"""Speech synthesis pipeline stage."""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Union

sys.path.append('/home/rosen/Project/StyleTTS2')

from exp2.pipeline.base_pipeline import PipelineStage
from exp2.config.model_config import ModelRegistry
from exp2.config.ablation_config import AblationVariant
from exp.inference_benchmark_models import inference_monoDiT, inference_Dit, inference_styletts2
from exp.inference_decoTTS import syn_speech_by_second_model as syn_speech_by_second_model_deco
from exp.inference_decoTTS import get_second_model as get_second_model_deco
from exp.exp_utils import copy_reference_speech
import shutil

class SynthesisStage(PipelineStage):
    """Stage 0: Synthesize speech using models."""

    def __init__(self, models: List[Union[str, AblationVariant]]):
        super().__init__(name="Speech Synthesis", enabled=True)
        self.models = models

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute speech synthesis for all models."""
        syn_texts = context["syn_texts"]
        syn_styles = context["syn_styles"]
        output_dir = context["output_dir"]
        style_name = context.get("style_syntex_name", "random")
        save_attn = context.get("save_attn", False)
        save_cond = context.get("save_cond", False)

        attn_results = {}
        psd_results = {}

        # Create reference directory and copy reference speech
        ref_dir = Path(output_dir) / "reference" / style_name
        ref_dir.mkdir(parents=True, exist_ok=True)
        copy_reference_speech(syn_styles, str(ref_dir))
        print(f"✓ Reference speech copied to {ref_dir}")

        for model_spec in self.models:
            # Handle both string model names and AblationVariant objects
            if isinstance(model_spec, AblationVariant):
                model_meta = ModelRegistry.get_variant(model_spec)
                model_name = model_spec.variant_name
                base_model = model_spec.base_model
            else:
                if model_spec == "reference":
                    continue
                model_meta = ModelRegistry.get(model_spec)
                model_name = model_spec
                base_model = model_spec

            print(f"\n--- Synthesizing with {model_name} ---")

            model_output_dir = Path(output_dir) / model_name / style_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Dispatch to appropriate inference function based on base model type
            if "monoDiT" in base_model:
                attn_dict = self._infer_monodit(
                    model_meta, syn_texts, syn_styles,
                    str(model_output_dir), save_attn, save_cond, context)
                attn_results[model_name] = attn_dict

            elif "decoditICL" in base_model:
                attn_dict, psd_dict = self._infer_decodit_icl(
                    model_meta, syn_texts, syn_styles,
                    str(model_output_dir), context
                )
                attn_results[model_name] = attn_dict
                psd_results[model_name] = psd_dict

            elif "decodit_cfm" in base_model:
                attn_dict, psd_dict = self._infer_decodit(
                    model_meta, syn_texts, syn_styles,
                    str(model_output_dir), save_attn, save_cond, context
                )
                attn_results[model_name] = attn_dict
                psd_results[model_name] = psd_dict

            elif "styletts2" in base_model:
                self._infer_styletts2(
                    model_meta, syn_texts, syn_styles,
                    str(model_output_dir), context
                )

            elif base_model == "DiT":
                self._infer_dit(
                    model_meta, syn_texts, syn_styles,
                    str(model_output_dir), context
                )

            else:
                print(f"⚠ Model {model_name} (base: {base_model}) not supported, skipping")
                continue

            print(f"✓ {model_name} synthesis completed")

        context["attn_results"] = attn_results
        context["psd_results"] = psd_results

        return context

    def _infer_monodit(self, model_meta, texts, styles, output_dir, save_attn, save_cond, context):
        """Run MonoDiT inference."""
        ckpt_path = str(model_meta.checkpoint_path)
        config_path = str(model_meta.config_path)

        attn_dict = inference_monoDiT(
            ckpt_path, config_path, texts, styles, output_dir,
            model_meta.inference_params,
            save_attn=save_attn,
            save_cond=save_cond
        )
        return attn_dict

    def _infer_decodit(self, model_meta, texts, styles, output_dir, save_attn, save_cond, context, reference_dir=""):
        """Run DecoDiT inference using InferenceAPI."""
        if len(reference_dir) > 0:
            ref_dir = os.path.join(os.path.dirname(output_dir), reference_dir)
            if not os.path.isdir(ref_dir):
                Path(ref_dir).mkdir(exist_ok=True, parents=True)

        from exp.inferenceAPI_bertFusion import InferenceAPI

        ckpt_path = str(model_meta.checkpoint_path)
        config_path = str(model_meta.config_path)

        # Initialize InferenceAPI
        api = InferenceAPI(
            model_name=model_meta.name,
            ckpt_path=ckpt_path,
            config_path=config_path
        )

        # Get inference parameters from model_meta (works for both regular models and variants)
        infer_params = model_meta.inference_params

        # Extract reference wav paths from styles
        # styles format: (spk, emo, txt, speech_path)
        ref_wav_paths = [style[3] for style in styles]

        # Synthesize all samples using batch inference
        for i, ref_style in enumerate(styles):
            spk, emo, ref_txt, speech_path = ref_style

            ## copy reference
            #if len(reference_dir) > 0:
            #    r_wav_f = f'spk{spk}_{emo}_ref{r_id}.wav'
            #    shutil.copy(speech_path, os.path.join(reference_dir, r_wav_f))

            # Create output filename
            ref_texts = context.get('ref_texts', [])
            if ref_txt not in ref_texts:
                ref_texts.append(ref_txt)
            context['ref_texts'] = ref_texts

            r_id = ref_texts.index(ref_txt)
            out_prefix = f'spk{spk}_{emo}_ref{r_id}_syn'

            # Synthesize with hierStyle enabled (consistent with training)
            api.synthesize_batch(
                texts,
                speech_path,
                ref_txt,
                out_dir=output_dir,
                out_wav_prefix=out_prefix,
                hierStyle=infer_params.get('hierStyle', True),
                drop_trend=infer_params.get('drop_trend', False),
                alpha=infer_params.get('alpha', 0.3),
                beta=infer_params.get('beta', 0.7),
                diffusion_steps=infer_params.get('diffusion_steps', 5),
                cfg_strength=infer_params.get('cfg_strength', 3),
                trend_strength=infer_params.get('trend_strength', 1.0)
            )

        # Return empty dicts for now (attention/psd saving can be added if needed)
        attn_dict = {}
        psd_dict = {}

        return attn_dict, psd_dict

    def _infer_decodit_icl(self, model_meta, texts, styles, output_dir, context):
        """Run ICL DecoDiT inference using InferenceAPIICL (CFMDecoderV4)."""
        from exp.inferenceAPI_bertFusion_icl import InferenceAPIICL

        ckpt_path   = str(model_meta.checkpoint_path)
        config_path = str(model_meta.config_path)
        infer_params = model_meta.inference_params

        api = InferenceAPIICL(
            model_name=model_meta.name,
            ckpt_path=ckpt_path,
            config_path=config_path,
        )

        ref_texts = context.get('ref_texts', [])
        for ref_style in styles:
            spk, emo, ref_txt, speech_path = ref_style

            if ref_txt not in ref_texts:
                ref_texts.append(ref_txt)
            context['ref_texts'] = ref_texts

            r_id       = ref_texts.index(ref_txt)
            out_prefix = f'spk{spk}_{emo}_ref{r_id}_syn'

            api.synthesize_batch(
                texts,
                speech_path,
                ref_txt,
                out_dir=output_dir,
                out_wav_prefix=out_prefix,
                beta=infer_params.get('beta', 0.7),
                diffusion_steps=infer_params.get('diffusion_steps', 5),
                cfg_strength=infer_params.get('cfg_strength', 3),
                drop_trend=infer_params.get('drop_trend', False),
                trend_strength=infer_params.get('trend_strength', 1.0),
                need_uv_mask=infer_params.get('need_uv_mask', True),
            )

        return {}, {}

    def _infer_styletts2(self, model_meta, texts, styles, output_dir, context):
        """Run StyleTTS2 inference."""
        ckpt_path = str(model_meta.checkpoint_path)
        config_path = str(model_meta.config_path)

        inference_styletts2(
            ckpt_path, config_path, texts, styles, output_dir,
            model_meta.inference_params
        )

    def _infer_dit(self, model_meta, texts, styles, output_dir, context):
        """Run DiT inference."""
        ckpt_path = str(model_meta.checkpoint_path)
        config_path = str(model_meta.config_path)

        inference_Dit(
            ckpt_path, config_path, texts, styles, output_dir,
            model_meta.inference_params
        )
