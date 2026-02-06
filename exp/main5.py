import shutil
import sys, os, yaml, json

from exp.exp_utils import convert_json_to_pd2_fine, save_attn_dict

sys.path.append('/home/rosen/Project/StyleTTS2')
os.chdir('/home/rosen/Project/StyleTTS2')

#sys.path.append('/home/rosen/Project/DrawSpeech_PyTorch')
#os.chdir('/home/rosen/Project/DrawSpeech_PyTorch')

import torch
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Literal
from exp_utils import (copy_ref_speech, copy_reference_speech, fine_adjust_configs, get_synStyle_from_file,
                       get_synText_from_file, renew_dict, combine_jsons, convert_json_to_pd, convert_json_to_pd2)
from vis_data_adaptor import convert_vis_psd_json, convert_attn_json, convert_attnEnh_json
from extract_psd import extract_psdave, extract_psd_fine_class2, extract_psd
from exp.visualization import vis_psd, vis_emo_crossAttn, vis_psd_enh, show_attn_map, show_two_attn_map, vis_mono_guide_mask, vis_dual_utmos_rmse
from exp.statcz_psd import statcz_psd_mcd, statcz_psd_fine_mcd, statcz_psd_mcd_fine_class2
from exp.exp_utils import save_json
from vis_data_adaptor import adapt_mix_2d_r_m1_m2, adapt_attn_2d_block_model
from vis2 import cst_attn_2d_block_model, cst_melattn_2d_type_model
#from infer_exp import infer_exp
from mel_config import MelConfig          ################ BE CAREFUL; Must same with draw?.yaml #########
from tqdm import tqdm
from exp_wer import evaluate_wer
from utilities.guide_mask import make_guided_attention_masks2
from utilities.vis import save_plot
from inference_benchmark_models import inference_monoDiT, inference_Dit, inference_styletts2
from exec_utmosv2 import run_utmos
from const_param import model_meta, model_infer_config
from inference_decoTTS import syn_speech_by_second_model as syn_speech_by_second_model_deco, get_second_model as get_second_model_deco

def main_ablation(
        syn_styles: List[Tuple[Any, Any, Any, Any, Any, Any]],
        synTexts: List[str],
        start_step: int,
        end_step: int,
        out_dir: str = "option",  # middle result
        ref_dir: str = "ref",
        mix_ref_pe_types = ["none"],
        mono_guide_deltas = [0.2, 0.5, 0.8],
        fuse_strength_gammas = [0.2, 0.4, 0.6],
        save_attn_json_file = False,
        save_cond = False,
        psd_level = "frame"
):
    # IN: ckpt, config, inf_args, hyper
    root_dir, ckpt, model_configs = model_meta["monoDiT"]
    ckpt, model_configs = os.path.join(root_dir, ckpt), os.path.join(root_dir, model_configs)

    ## OUT (wav): out_dir, attn_dir, ref_dir attn_json_path
    attn_json_path = os.path.join(out_dir, "attn_mdit_random.json")  # only needed in random synthesis
    psdcond_json_path = os.path.join(out_dir, "psdcond_mdit_random.json")  # only needed in random synthesis
    out_speech_dir = os.path.join(out_dir, "monoDiT_ablation")  # out_dir/model_name/[random/CondA/CondB]]/...

    # Synthesis speech
    if start_step <= 0 <= end_step:
        attn_dict, psdcond_dict = inference_monoDiT(ckpt, model_configs, synTexts, syn_styles, out_speech_dir,
                                      model_infer_config["monoDiT"], mix_ref_pe_types=mix_ref_pe_types,
                                      mono_guide_deltas=mono_guide_deltas, fuse_strength_gammas=fuse_strength_gammas,
                                      save_attn=save_attn_json_file, save_cond=save_cond)

        if save_attn_json_file and len(attn_dict) != 0:
            with open(attn_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(attn_dict, sort_keys=True, indent=4))
        if save_cond:
            with open(psdcond_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(psdcond_dict, sort_keys=True, indent=4))

    prosody_dict = {}
    psd_statics_json = os.path.join(out_dir, f"stats_psd_ablation.json")
    psd_statics_mean_json = os.path.join(out_dir, f"stats_psd_ablation_mean.json")

    # OUT (paper related):  multiIndex_psd, multiIndex_meanModel, wer_utmos
    statsave_psd = os.path.join(out_dir, "statsIntp_psd_mulitindex_ablation.csv")
    statsave_psd_mean = os.path.join(out_dir, "statsIntp_psd_mulitindex_meanModel_ablation.csv")
    wer_utmos_json_path = os.path.join(out_dir, "wer_utmos_ablatoin.json")

    # Extract psd and compute stats
    if start_step <= 1 <= end_step:
        print("Start extract frame level pitch/energy from {} dir".format(out_speech_dir))
        if not os.path.isdir(ref_speech_dir):
            Path(ref_speech_dir).mkdir(exist_ok=True, parents=True)
        copy_reference_speech(syn_styles, ref_dir)

        # extract psd
        if psd_level == "frame":
            psd_json_path = os.path.join(out_dir, "psd_ablation.json")   # OUT
            for i, ablation_name in enumerate(os.listdir(out_speech_dir)):
                if i == 0:
                    prosody_dict = extract_psd(mel_config, ref_dir, model_n="reference", save_psd_file="", prosody_dict=prosody_dict)
                out_speech_ablation_dir = os.path.join(out_speech_dir, ablation_name)
                if not os.path.isdir(out_speech_ablation_dir):
                    continue
                prosody_dict = extract_psd(mel_config, out_speech_ablation_dir, model_n=ablation_name, save_psd_file="",
                                           prosody_dict=prosody_dict)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
        elif psd_level == "phoneme":   ######## NOT USED currently
            psd_json_path = os.path.join(out_dir, "psdave_ablation.json") # OUT
            cmp_modelnames = [ablation_name for ablation_name in os.listdir(out_speech_dir) if not ablation_name.endswith("attn")]
            prosody_psdave_dict = extract_psdave(mel_config, cmp_modelnames, out_dir)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_psdave_dict, sort_keys=True, indent=4))
        else:
            raise IOError("psd level error!")

        # stats psd
        with open(psd_json_path, "r") as f:
            prosody_dict = json.load(f)
        psd_ctw_res, psd_mcd_stat_res, psd_mean_stat_res = statcz_psd_mcd(prosody_dict, exclude_zero=True)  # {"spk": {"ang": {"modelA": [p1, e1, m1]}, "happy":{ "modelB": []]}}}
        with open(psd_statics_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mcd_stat_res, sort_keys=True, indent=4))
        with open(psd_statics_mean_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mean_stat_res, sort_keys=True, indent=4))

        # paper related output format
        psd_mean_stat_res_pd, mean_df = convert_json_to_pd2(psd_mean_stat_res)  # model     hyper     pitch    energy
        psd_mean_stat_res_pd.to_csv(statsave_psd, index=True)
        mean_df.to_csv(statsave_psd_mean, index=True)
        print(psd_mean_stat_res_pd)

    # Wer computing
    if start_step <= 2 <= end_step:
        wer_utmos2_dict = {}
        for i, ablation_name in enumerate(os.listdir(out_speech_dir)):
            out_speech_ablation_dir = os.path.join(out_speech_dir, ablation_name)  # out_dir/model_name/[random/CondA/CondB]]/...
            if not os.path.isdir(out_speech_ablation_dir) or out_speech_ablation_dir.endswith("_attn"):
                continue
            wer, sub, dele, ins = evaluate_wer(out_speech_ablation_dir, output_csv=f"{out_speech_dir}/wer_{ablation_name}.csv")
            mean_mos, std_mos, mos_list = run_utmos(out_speech_ablation_dir, 1, output_file=f"{out_speech_dir}/utmos_{ablation_name}.txt")
            wer_utmos2_dict[ablation_name] = [wer, mean_mos]
            print(f"wer/sub/dele/ins of {ablation_name} are: {wer}, {sub}, {dele}, {ins}")
            print(f"{ablation_name} UTMOS-v2: {mean_mos:.4f} ± {std_mos:.4f}")
        with open(wer_utmos_json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(wer_utmos2_dict, sort_keys=True, indent=4))
        print(wer_utmos2_dict)

def main(
        syn_styles: List[Tuple[Any, Any, Any, Any, Any, Any]],
        synTexts: List[str],
        cmp_modelnames: List[str],
        ref_json: str,
        out_dir: str = "option",  # middle result
        start_step=1,
        end_step=2,
        mel_config=None,
        vis_attn_config=None,
        vis_psd_config=None,
        vis_psdmel_config=None,
        save_attn_json_file=True,
        style_syntex_name="random",
        infer_json_name="infer.json",
        psd_level="phoneme",
        save_attn=True,
        save_cond=False,
        save_attn_time=5,
    ):
    """
    0: Syn speech, save attn, copy ref, create attn_json
    1: extract psd_json
    2: extract SIM-O and SIM-R  (?)
    3: vis attention
    4: vis psd
    5: compute statistics

    IN:
        - syn_styles, synText, cmp_modelnames
    OUT:
        - speech_dir|syn_ref_type (with [pair_type]_attn).   <- speech_dir
            - speech_dir: lddpm_dit_pe_libritts, ...
            - syn_ref_type: random, lenRatio, pos
        - ref_dir|random
        - infer.json
        - attn_modelA_modelB_ref.json
        - psd_modelA_ref.json
        - stats_modelA.json

    ref_json: used to provide meta info
    """
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir(exist_ok=True, parents=True)
    # Create infer.json, which is fed to dataset model for synthesis. IN:
    infer_json_path = os.path.join(out_dir, infer_json_name)
    gd_speech_dir = os.path.join(out_dir, "gd_speech")
    if not os.path.isdir(gd_speech_dir):
        Path(gd_speech_dir).mkdir(exist_ok=True, parents=True)

    #res_dir = os.path.join(out_dir, "res")
    #if not os.path.isdir(res_dir):
    #    Path(res_dir).mkdir(exist_ok=True, parents=True)

    ####### 0: syn text by single moddel (preprare) #######
    if start_step <= 0 <= end_step:
        for i, model_name in enumerate(cmp_modelnames):
            # IN: ckpt, config, inf_args, hyper
            if model_name in ["drawspeech", "hierspeech"]:
                continue
            root_dir, ckpt, model_configs = model_meta[model_name]
            ckpt, model_configs = os.path.join(root_dir, ckpt), os.path.join(root_dir, model_configs)

            ## OUT: out_dir, attn_dir, ref_dir attn_jason_path
            out_speech_dir = os.path.join(out_dir, model_name, style_syntex_name)  #out_dir/model_name/[random/CondA/CondB]]/...
            out_attn_dir = os.path.join(out_dir, model_name, style_syntex_name + "_attn")  # few attn is enough?
            ref_speech_dir = os.path.join(out_dir, "reference", style_syntex_name)  #
            attn_json_path = os.path.join(out_dir, "attn_{}_{}.json".  # {spk/emo/model: {speech_id:[], syn_phonemes:[], ref_phonemes:[], q_dur:[], k_dur:[]}}
                                          format(model_name, style_syntex_name))  # only needed in random synthesis
            psdcon_json_path = os.path.join(out_dir, "psdcond_{}_{}.json".  # {spk/emo/model: {speech_id:[], syn_phonemes:[], ref_phonemes:[], q_dur:[], k_dur:[]}}
                                          format(model_name, style_syntex_name))  # only needed in random synthesis

            if not os.path.isdir(out_speech_dir):
                Path(out_speech_dir).mkdir(exist_ok=True, parents=True)
            if not os.path.isdir(ref_speech_dir):
                Path(ref_speech_dir).mkdir(exist_ok=True, parents=True)
            if not os.path.isdir(out_attn_dir) and save_attn and "dit" in model_name:
                Path(out_attn_dir).mkdir(exist_ok=True, parents=True)
            if i == 0:
                copy_reference_speech(syn_styles, ref_speech_dir)
                #copy_gd_speech(syn_styles)

            # syn speech, save attn, create attn_json (for exp)
            if "monoDiT" in model_name:
                print("synthesized by monoDiT model")
                attn_dict = inference_monoDiT(ckpt, model_configs, synTexts, syn_styles, out_speech_dir,
                                              model_infer_config[model_name], save_attn=save_attn, save_cond=save_cond)
                # attn_json_path for showing attention
                if save_attn_json_file and len(attn_dict) != 0:
                    with open(attn_json_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(attn_dict, sort_keys=True, indent=4))
            elif "deco" in model_name:
                print("synthesized by decoDiT model")

                second_model, sampler, model_params = get_second_model_deco(ckpt=ckpt, config_f=model_configs, model_name="mdit_cfm")
                attn_dict, psdcond_json = syn_speech_by_second_model_deco(
                    synTexts, syn_styles, out_speech_dir, second_model, sampler,  # input, output, model
                    **model_infer_config[model_name],     # model config
                    save_attn=save_attn, save_cond=save_cond, model_name=model_name)  # output type

                if save_attn_json_file and len(attn_dict) != 0:
                    save_json(attn_dict, attn_json_path)
                if save_cond and len(psdcond_json) != 0:
                    save_json(psdcond_json, psdcon_json_path)
            elif "styletts2" in model_name:
                print("synthesized by styletts2 model")
                inference_styletts2(ckpt, model_configs, synTexts, syn_styles, out_speech_dir,
                                              model_infer_config[model_name])
            elif model_name == "DiT":
                print("synthesized by DiT model")
                inference_Dit(ckpt, model_configs, synTexts, syn_styles, out_speech_dir,
                                              model_infer_config[model_name])
            else:
                print(f"{model_name} not supported!!")
    else:
        print("1. Skip speech synthesis!")

    ####### 1: extract psd from speech folder#######
    if start_step <= 1 <= end_step:
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        prosody_dict = {}
        if psd_level == "frame":
            psd_json_path = os.path.join(out_dir, "psd_{}.json".format("_".join(cmp_modelnames)))
            for model_name in cmp_modelnames:
                print("Start extract {} level pitch/energy from {} dir".format(psd_level, model_name))
                out_speech_dir = os.path.join(out_dir, model_name, style_syntex_name)
                prosody_dict = extract_psd(mel_config, out_speech_dir, model_n=model_name, save_psd_file="", prosody_dict=prosody_dict)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
        elif psd_level == "phoneme":
            psd_json_path = os.path.join(out_dir, "psdave_{}.json".format("_".join(cmp_modelnames)))
            prosody_psdave_dict = extract_psdave(mel_config, cmp_modelnames, out_dir)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_psdave_dict, sort_keys=True, indent=4))

    ####### 2. Statistics psd and mc given psd_{}.json (output of 1) #######
    statsave_psd = os.path.join(out_dir, "statsIntp_psd_mulitindex.csv")
    statsave_psd_mean = os.path.join(out_dir, "statsIntp_psd_mulitindex_meanModel.csv")
    if start_step <= 2 <= end_step:
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        cmp_modelnames_combine = "_".join(cmp_modelnames)
        if psd_level == "frame":
            psd_json_path = os.path.join(out_dir, "psd_{}.json".format(cmp_modelnames_combine))
            psd_statics_json = os.path.join(out_dir, f"statsIntp_psd_{cmp_modelnames_combine}.json")
            psd_ctw_res_json = os.path.join(out_dir, f"dtwIntp_psd_{cmp_modelnames_combine}.json")
            psd_statics_mean_json = os.path.join(out_dir, f"statsIntp_psd_{cmp_modelnames_combine}_mean.json")  # mean on speakers
        elif psd_level == "phoneme":
            psd_json_path = os.path.join(out_dir, "psdave_{}.json".format(cmp_modelnames_combine))
            psd_statics_json = os.path.join(out_dir, f"statsave_psd_{cmp_modelnames_combine}.json")
            psd_ctw_res_json = os.path.join(out_dir, f"ctw_psd_{cmp_modelnames_combine}.json")
            psd_statics_mean_json = os.path.join(out_dir, f"statsave_psd_{cmp_modelnames_combine}_mean.json")
        else:
            raise IOError(f"{psd_level} is not supported!!")
        with open(psd_json_path, "r") as f:
            prosody_dict = json.load(f)
        psd_ctw_res, psd_mcd_stat_res, psd_mean_stat_res = statcz_psd_mcd(prosody_dict, exclude_zero=True)  # {"spk": {"ang": {"modelA": [p1, e1, m1]}, "happy":{ "modelB": []]}}}
        with open(psd_ctw_res_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_ctw_res, sort_keys=True, indent=4))
        with open(psd_statics_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mcd_stat_res, sort_keys=True, indent=4))
        with open(psd_statics_mean_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mean_stat_res, sort_keys=True, indent=4))

        psd_mean_stat_res_pd, mean_df = convert_json_to_pd2(psd_mean_stat_res,
                                                   need_multi_index=True, need_print_latex=True)    # model     hyper     pitch    energy
        psd_mean_stat_res_pd.to_csv(statsave_psd, index=True)
        mean_df.to_csv(statsave_psd_mean, index=True)

        # angry: hier, style, dit, mono
        # happy: Dit, mono
        # neutral: hier, style, mono
        # sad: hier, style, mono
        # supprise: hier, mono, dit, draw, style
        print(psd_mean_stat_res_pd)

        # {"spk": {"ang": {"modelA": {"04": [p1, e1, m1]}}}}   <-02 06 10 14 18 22
        # {"spk": {"ang": {"modelA": {"spos": [p1, e1, m1]}}   <-spos, dpos
        # convert json to pandas for paper writing

    ####### 3: WER and UtmosV2 #######
    wer_utmos2_dict = {}
    wer_utmos_json_path = os.path.join(out_dir, "wer_utmos.json")
    if start_step <= 3 <= end_step:
        for model_name in cmp_modelnames:
            out_speech_dir = os.path.join(out_dir, model_name, style_syntex_name)  #out_dir/model_name/[random/CondA/CondB]]/...
            wer, sub, dele, ins = evaluate_wer(out_speech_dir, output_csv=f"{out_dir}/wer_{model_name}.csv")
            mean_mos, std_mos, mos_list = run_utmos(out_speech_dir, 1)
            print(f"wer/sub/dele/ins of {model_name} are: {wer}, {sub}, {dele}, {ins}")
            print(f"{model_name} UTMOS-v2: {mean_mos:.4f} ± {std_mos:.4f}")
            wer_utmos2_dict[model_name] = [wer, sub, dele, ins, mean_mos, std_mos]
            with open(wer_utmos_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(wer_utmos2_dict, sort_keys=True, indent=4))

        # do wer on gd_speech
        if not os.path.isfile(f"{out_dir}/wer_gd_speech.csv"):
            out_speech_dir = os.path.join(out_dir, "gd_speech")
            wer, sub, dele, ins = evaluate_wer(out_speech_dir, output_csv=f"{out_dir}/wer_gd_speech.csv")
            mean_mos, std_mos, mos_list = run_utmos(out_speech_dir, 1)
            print(f"wer/sub/dele/ins of gd_speech are: {wer}, {sub}, {dele}, {ins}")
            print(f"GD UTMOS-v2: {mean_mos:.4f} ± {std_mos:.4f}")

    ####### 4. Vis psd contour given json file  #######
    img_out = os.path.join(out_dir, "img_out1")
    if not os.path.isdir(img_out):
        Path(img_out).mkdir(exist_ok=True, parents=True)

    # save pitch_dict_for_vis
    if start_step <= 4 <= end_step and len(cmp_modelnames) > 1:
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        cmp_modelnames_combine = "_".join(cmp_modelnames)
        if psd_level == "frame":
            psd_json_path = os.path.join(out_dir, "psd_{}.json".format("_".join(cmp_modelnames)))
        else:
            psd_json_path = os.path.join(out_dir, "psdave_{}.json".format("_".join(cmp_modelnames)))
        # psd_json_path = "/hdd/StableTTS/exp/cfm_dstgl/psdave_cfm_dit_cross_distgl_cfm_mdit_cross_distgl_styletts2_reference.json"
        with open(psd_json_path, "r") as f:
            psd_cmp_dict = json.load(f)
        show_spk, show_text = vis_psd_config["show_spk"], vis_psd_config["show_txt"]
        psd_compare_png = os.path.join(img_out, f"psd_{cmp_modelnames_combine}_ref{show_text[0]}_syn{show_text[1]}.png")
        pitch_cmp_path, energy_cmp_path = (os.path.join(out_dir, f"vis_pitch_{cmp_modelnames_combine}.json"),
                                           os.path.join(out_dir, f"vis_energy_{cmp_modelnames_combine}.json"))
        # visualize psd
        pitch_dict_for_vis, energy_dict_for_vis = convert_vis_psd_json(
            psd_cmp_dict["spk" + show_spk], show_text, cutpad_reference=False, save_dict=(pitch_cmp_path, energy_cmp_path))  # for_vis: {"emo1": {"model1": list(psd_len)}}}
        vis_psd(pitch_dict_for_vis, energy_dict_for_vis, psd_compare_png, show_text, ordered_lengend=orderd_cmp_modelnames)

    ######## 5. Vis mono maps: vis mono-guidance matrix on different strength (x) /text_length (y)
    if start_step <= 5 <= end_step:
        dul_axis_data = {"y1": [40, 50, 60], "y2": [3.01, 3.22, 3.21], "x": [0.2, 0.5, 0.8]}
        dual_rmse_utmos_png = os.path.join(img_out, f"vis_dual_rmse_utmos.pdf")
        vis_dual_utmos_rmse(dul_axis_data, out_pic=dual_rmse_utmos_png)

        guid_masks_png = os.path.join(img_out, f"vis_guid_masks.pdf")
        # draw row 3 * column 2 images, given x=simga=[[0.2, 0.5, 0.8]], y=mel_len=[100, 200]
        mono_guide_pics = {"x": [0.2, 0.5, 0.8], "y": [100, 200]}

        vis_mono_guide_masks_pngs = []
        for j, max_len in enumerate(mono_guide_pics["y"]):
            for i, sigma in enumerate(mono_guide_pics["x"]):
                ilens = torch.tensor([max_len]).unsqueeze(0)
                olens = torch.tensor([max_len]).unsqueeze(0)
                guide_matrix2 = make_guided_attention_masks2(ilens, olens, max_len=max_len, base_sigma=sigma, eps=0.002)
                vis_mono_guide_masks_png = os.path.join(img_out, f"vis_mono_masks_x{i}_y{j}.png")
                vis_mono_guide_masks_pngs.append(vis_mono_guide_masks_png)
                save_plot(guide_matrix2[0].detach().cpu(), vis_mono_guide_masks_png)
        vis_mono_guide_mask(vis_mono_guide_masks_pngs, guid_masks_png)

    ####### 6. Vis attention map given attn file #######
    if start_step <= 6 <= end_step and len(cmp_modelnames) > 1:
        # combine attn_json
        attn_model_paths = [os.path.join(out_dir, "attn_{}_random.json".format(model_name)) for model_name in
                            cmp_modelnames if "dit" in model_name]
        cmp_modelnames_combine = "_".join(cmp_modelnames)
        attn_cmp_path = os.path.join(out_dir, f"attn_{cmp_modelnames_combine}.json")
        if len(attn_model_paths) == 2:
            combine_jsons(attn_model_paths[0], attn_model_paths[1], attn_cmp_path)
        else:
            IOError("more than 3 combination is not supported!")
        with open(attn_cmp_path, "r") as f:
            attn_dict = json.load(f)
        t, h, txt, emo, tg = vis_attn_config["show_t"], vis_attn_config["show_h"], vis_attn_config["show_txt"], \
        vis_attn_config["show_emo"], vis_attn_config["tick_gran"]
        show_spk = vis_attn_config["show_spk"]
        # vis attn
        adpt_attn_data, png_n, title = adapt_attn_2d_block_model(attn_dict["spk" + show_spk], out_dir, show_t=t,
                                                                 show_h=h, show_txt=txt, show_emo=emo, tick_gran=tg,
                                                                 model_ab=(cmp_modelnames[0], cmp_modelnames[1]))  # phoneme
        cst_attn_2d_block_model(adpt_attn_data, png_n, title)
        # vis attn&mel
        adpt_attn_mel_data, png_n, title = adapt_mix_2d_r_m1_m2(attn_dict["spk" + show_spk], out_dir, show_t=t,
                                                                show_h=h, show_txt=txt, show_emo=emo, tick_gran=tg,
                                                                model_ab=(cmp_modelnames[0], cmp_modelnames[1]))  # syllable
        cst_melattn_2d_type_model(adpt_attn_mel_data, png_n, title)

def add_emo(eval_style_f, meta_json, eval_style_emo_f):
    """
    add emotion to eval_style file
    """
    with open(meta_json) as f:
        meta_dict = json.load(f)
    eval_style_emo_list = []
    with open(eval_style_f) as f:
        for l in f:
            speechid = l.split("|")[0]
            emo = meta_dict[speechid]["emotion"]
            eval_style_emo_list.append(l[:-1] + "|" + emo + "\n")
    with open(eval_style_emo_f, "w") as f:
        f.writelines(eval_style_emo_list)

def main_fine_analysis(start_step, end_step, cmp_modelnames, style_syntex_names, out_dir):
    ####### 1: extract psd from speech folder#######
    if start_step <= 1 <= end_step:
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        # OUT
        prosody_dict = {}
        psd_json_path = os.path.join(out_dir, "psd_{}.json".format("_".join(cmp_modelnames)))

        for model_name in cmp_modelnames:
            for style_syntex_name in style_syntex_names:
                print("Start extract frame-level pitch/energy from {} dir".format(model_name))
                # IN
                out_speech_dir = os.path.join(out_dir, model_name, style_syntex_name)
                prosody_dict = extract_psd_fine_class2(mel_config, out_speech_dir, model_n=model_name,
                                                       save_psd_file="", prosody_dict=prosody_dict, fine_cate=style_syntex_name)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
        with open(psd_json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))

    ####### 2. Statistics psd and mc given psd_{}.json (output of 1) #######
    if start_step <= 2 <= end_step:
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        cmp_modelnames_combine = "_".join(cmp_modelnames)
        # IN
        psd_json_path = os.path.join(out_dir, "psd_{}.json".format(cmp_modelnames_combine))

        # OUT
        psd_statics_json = os.path.join(out_dir, f"statsIntp_psd_{cmp_modelnames_combine}.json")
        psd_ctw_res_json = os.path.join(out_dir, f"dtwIntp_psd_{cmp_modelnames_combine}.json")
        psd_statics_mean_json = os.path.join(out_dir,
                                             f"statsIntp_psd_{cmp_modelnames_combine}_mean.json")  # mean on speakers
        statsave_pitch_mean = os.path.join(out_dir, "statsIntp_psd_mulitindex_pitch.csv")
        statsave_energy_mean = os.path.join(out_dir, "statsIntp_psd_mulitindex_energy.csv")

        with open(psd_json_path, "r") as f:
            prosody_dict = json.load(f)
        psd_ctw_res, psd_mcd_stat_res, psd_mean_stat_res = statcz_psd_mcd_fine_class2(prosody_dict, exclude_zero=True)  # {"spk": {"ang": {"modelA": [p1, e1, m1]}, "happy":{ "modelB": []]}}}
        with open(psd_ctw_res_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_ctw_res, sort_keys=True, indent=4))
        with open(psd_statics_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mcd_stat_res, sort_keys=True, indent=4))
        with open(psd_statics_mean_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mean_stat_res, sort_keys=True, indent=4))
        pivot_pitch, pivot_energy = convert_json_to_pd2_fine(psd_mean_stat_res)  # model     hyper     pitch    energy

        pivot_pitch.to_csv(statsave_pitch_mean, index=True)
        pivot_energy.to_csv(statsave_energy_mean, index=True)

        # angry: hier, style, dit, mono
        # happy: Dit, mono
        # neutral: hier, style, mono
        # sad: hier, style, mono
        # supprise: hier, mono, dit, draw, style
        print(pivot_pitch)


if __name__ == "__main__":
    """
    -1: Generate infer.json (3Type) from given style and text file
    0: Synthesize speech dir for each model and generate attn_json
    1: extract psd_json
    2: CTW
    3: WER, UTMOS_v2
    4. vis psd
    5: vis attn/mel
    On doing: extract SIM-O and SIM-R  (?)
    """
    seed = 0
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()

    ##### 1. CONSTANTS:
    dataset_name_rootdir = {"esd": "/hdd/ESD", "libritts": "LibriTTS_16k"}
    meta_data_dir = "/home/rosen/Project/DrawSpeech_PyTorch/data/dataset/metadata"
    ref_json = {
        "esd": os.path.join(meta_data_dir, "esd/train.json"),
        "libritts": os.path.join(meta_data_dir, "libritts16k_cutdur/train.json")}
    #mname_config = lambda config_name: os.path.join("drawspeech/config", config_name)
    mname_config = {
        "mdit_librittsesd_cutdurspn_pe": "drawspeech/config/mdit_librittsesd_cutdurspn_pe_infer.yaml"}
    OTHER_CMP_MODELS = ["styletts2", "natrualspeech2"]


    ##### 2. EXP CONFIG
    mel_config = MelConfig
    # show_spks, show_emos, show_texts = [("0019", "Surprise", 0), ("0013", "Angry", 0), ("0017", "Surprise", 3)]   # spk0019_Surprise_ref3_syn3
    # spk0019_Surprise_ref3: he was still in the forest!
    vis_attn_config = {
        "esd": {"show_spk": "0019", "show_t": 0, "show_h": 0, "show_txt": 3, "show_emo": "Surprise",
                "tick_gran": "syllable"},
        "libritts": {"show_spk": "?", "show_t": 0, "show_h": 0, "show_txt": 3, "show_emo": "Neutral",
                     "tick_gran": "syllable"}}
    vis_psd_config = {
        "esd": {"show_spk": "0019", "show_emo": "Surprise", "tick_gran": "syllable", "show_txt": (4, 3)},
        "libritts": {"show_spk": "0019", "show_emo": "Surprise", "tick_gran": "syllable", "show_txt": (3, 4)}}  # ref3:  15 - 19
    vis_psdmel_config = {
        "esd": {"show_spk": "0013", "show_emo": "Surprise", "tick_gran": "syllable"},
        "libritts": {"show_spk": "0013", "show_emo": "Surprise", "tick_gran": "syllable"}}

    ######### 3. INPUT
    parser.add_argument("--style", type=str, default="exp/data/r1_50.txt")  # evalstyle  r1_test  r1 libri_r1, r1_v2, r1_50, r1_test
    parser.add_argument("--txt", type=str, default="exp/data/s1_5.txt")  # evaltxt_para.txt libri_s1, s1_v2, s1_5, s1_2
    dataset_name = "esd"    # libritts esd

    args = parser.parse_args()
    # orderd_cmp_modelnames = ["reference", "styletts2", "lddpm_dit_pe_libritts"]
    #orderd_cmp_modelnames = ["reference", "styletts2", "mdit_librittsesd_cutdurspn_pe"]
    #orderd_cmp_modelnames = ["reference", "hierspeech", "monoDiT"]
    orderd_cmp_modelnames = ["reference", "hierspeech", "styletts2", "monoDiT"]

    #eval_models = ["monoDiT", "DiT", "drawspeech", "styletts2", "hierspeech"]  # "monoDiT", "DiT", "drawspeech", "styletts2", "hierspeech"
    #eval_models = ["monoDiT_ab0808_m08_fb03", "monoDiT_ab0307_m08_fb03", "monoDiT_ab0808_m08_fbnone"]
    eval_models = ["monoDiT", "DiT", "drawspeech", "styletts2", "hierspeech"]
    eval_models = ["monoDiT"]

    # INPUT -> syn_styles (emo, spk, wav_p, psd_code), synTexts, and vis related ()
    #TEST_PART_NUM = 2
    ### CHIMPION: he was still in the forest! (high pitch) <- spk19. txt0
    # eval_models = ["drawspeech_libritts_16k_spk_cutdur", "drawspeech_libritts_mdit_16k_cutdur_phase2"] # ["cfm_dit_self", "cfm_dit_cross_distgl", "cfm_mdit_cross_distgl", "styletts2"]

    ######### OUTPUT
    out_dir = f"/home/rosen/ckpt/exp/mdit_tts_{dataset_name}"   # esd, _mdit_multiversion
    syn_styles = get_synStyle_from_file(args.style, split_char='|', melstyle_type="codec", dataset_name=dataset_name)  # emotion changed
    synTexts = get_synText_from_file(args.txt)

    EVAL_RANDOM = False
    EVAL_ABLATION = False
    EVAL_FINE1 = True
    EVAL_FINE2 = False
    if EVAL_RANDOM:
        NUM = 1
        for i in range(NUM):
            for j in range(NUM):
                vis_psd_config[dataset_name]["show_txt"] = (i, j)   # cmp(ref,syn) on psd contours = (4, 2), (4, 0)
                main(
                    syn_styles=syn_styles,
                    synTexts=synTexts,
                    cmp_modelnames=eval_models,
                    ref_json=ref_json[dataset_name],
                    out_dir=out_dir,
                    start_step=1,
                    end_step=1,
                    mel_config=mel_config,
                    vis_attn_config=vis_attn_config[dataset_name],
                    vis_psd_config=vis_psd_config[dataset_name],
                    vis_psdmel_config=vis_psdmel_config[dataset_name],
                    save_attn_json_file=True,
                    style_syntex_name="random",
                    infer_json_name="infer.json",
                    psd_level="phoneme",
                    save_attn=True
                )

    if EVAL_ABLATION:
        # v2:????(attn affect pitch) v4: "train_epoch48" <- base, v5: "train_epoch68", v6: "train_w/_monoGuide",
        # v7: "self_mask"
        MONO_ABL = True
        FUSE_ABL = False
        if MONO_ABL:
            out_dir = f"/home/rosen/ckpt/exp/mdit_tts_{dataset_name}_ablation_mono_v7"  # esd, _mdit_multiversion
            ref_speech_dir = os.path.join(out_dir, "reference", "random")
            main_ablation(
                syn_styles=syn_styles,
                synTexts=synTexts,
                start_step=0,
                end_step=0,
                out_dir=out_dir,
                ref_dir=ref_speech_dir,
                mix_ref_pe_types=["none"],  # ref_pred_add none
                mono_guide_deltas=[0.2, -1.0],  # 0.2, 0.5, 0.8, -1.0
                fuse_strength_gammas=[0.2],
                save_attn_json_file=True,
                save_cond=True)
        if FUSE_ABL:
            out_dir = f"/home/rosen/ckpt/exp/mdit_tts_{dataset_name}_ablation_fuse"  # esd, _mdit_multiversion
            ref_speech_dir = os.path.join(out_dir, "reference", "random")
            # 1. ref_pe: reference pitch/energy (pe)
            # 2. none  : predicted pe
            # 3. ref_pred_add: fusing reference pe to predicted pe
            main_ablation(
                syn_styles=syn_styles,
                synTexts=synTexts,
                out_dir=out_dir,
                ref_dir=ref_speech_dir,
                mix_ref_pe_types=["ref_pe", "none", "ref_pred_add"],
                mono_guide_deltas=[0.5],
                fuse_strength_gammas=[0.2, 0.4, 0.6],
                save_attn_json_file=False,
                save_cond=True)
    if EVAL_FINE1:
        """
        v1
        """
        # Test on synText and reference speech which are on two restriction
        ## 1. ? < syn_l / ref_l < ?
        ## 2. parts_speech(syn_txt) != parts_speech(ref_text)
        ## INPUT
        style_syn_f_dir = "/home/rosen/Project/StyleTTS2/exp/data"
        ref_style_f = os.path.join(style_syn_f_dir, "r1_50.txt")  # should be r2, but for more data, use r1.txt
        #eval_models = ["monoDiT", "monoDiT_ab0307", "decoDiT_v16"]  # "monoDiT", "DiT", "drawspeech", "styletts2", "hierspeech"
        eval_models = ["monoDiT", "monoDiT_ab0307", "decoDiT_v16"]
        eval_models = ["monoDiT_ab0307"]

        # eval_models = ["monoDiT", "DiT", "drawspeech", "styletts2", "hierspeech"]  # "monoDiT", "DiT", "drawspeech", "styletts2", "hierspeech"
        ## OUTPUT
        out_dir = f"/home/rosen/ckpt/exp/mdit_tts_{dataset_name}_fine1_multiversion"  #
        fine_categ_labels = ["2"]  # ["05", "1", "2"]
        fine_categ_fs = [os.path.join(style_syn_f_dir, "s2_" + fine_cat + ".txt") for fine_cat in fine_categ_labels]
        syn_styles = get_synStyle_from_file(ref_style_f, split_char='|', melstyle_type="codec")  # emotion changed
        fine_synTexts = [get_synText_from_file(fine_categ_f) for fine_categ_f in fine_categ_fs]
        if True:
            for fine_cate_label, fine_synText in zip(fine_categ_labels, fine_synTexts):
                main(
                    syn_styles=syn_styles,  # syn_styles[0:25:5] print one sample per emo of spk19
                    synTexts=fine_synText,
                    cmp_modelnames=eval_models,
                    ref_json=ref_json[dataset_name],
                    out_dir=out_dir,
                    start_step=1,
                    end_step=2,    # Set (start_step, end_step)=(0,0), and then set to (1, 2) with break below
                    mel_config=mel_config,
                    vis_attn_config=vis_attn_config[dataset_name],
                    vis_psd_config=vis_psd_config[dataset_name],
                    vis_psdmel_config=vis_psdmel_config[dataset_name],
                    save_attn_json_file=True,
                    style_syntex_name=fine_cate_label,
                    infer_json_name=f"infer_{fine_cate_label}.json",
                    psd_level="frame",
                    save_attn=False)
                break # break at second 1, 2
        #main_fine_analysis(2, 2, eval_models, fine_categ_labels, out_dir)

    if EVAL_FINE2:
        ## INPUT
        style_syn_f_dir = "/home/rosen/Project/StyleTTS2/exp/data"
        ref_style_f = os.path.join(style_syn_f_dir, "r2txt")
        eval_models = ["monoDiT", "DiT", "drawspeech", "styletts2", "hierspeech"]  # "monoDiT", "DiT", "drawspeech", "styletts2", "hierspeech"

        ## OUTPUT
        out_dir = f"/home/rosen/ckpt/exp/mdit_tts_{dataset_name}_fine2_multiversion"  #

        fine_categ_labels = ["same_pos", "diff_pos"]
        fine_categ_fs = [os.path.join(style_syn_f_dir, "s3_" + fine_cat + ".txt") for fine_cat in fine_categ_labels]
        syn_styles = get_synStyle_from_file(ref_style_f, split_char='|', melstyle_type="codec")  # emotion changed
        fine_synTexts = [get_synText_from_file(fine_categ_f) for fine_categ_f in fine_categ_fs]

        if False:
            for fine_cate_label, fine_synText in zip(fine_categ_labels, fine_synTexts):
                main(
                    syn_styles=syn_styles,
                    synTexts=fine_synText,
                    cmp_modelnames=eval_models,
                    ref_json=ref_json[dataset_name],
                    out_dir=out_dir,
                    start_step=1,
                    end_step=2,  # DO first 0,0, and second 1, 2
                    mel_config=mel_config,
                    vis_attn_config=vis_attn_config[dataset_name],
                    vis_psd_config=vis_psd_config[dataset_name],
                    vis_psdmel_config=vis_psdmel_config[dataset_name],
                    save_attn_json_file=True,
                    style_syntex_name=fine_cate_label,
                    infer_json_name=f"infer_{fine_cate_label}.json",
                    psd_level="frame",
                    save_attn=False)
        main_fine_analysis(2, 2, eval_models, fine_categ_labels, out_dir)