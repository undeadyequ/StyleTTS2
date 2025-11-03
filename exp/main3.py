import sys, os, yaml, json
sys.path.append('/home/rosen/Project/StableTTS')

from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Literal
from exp.syn_speech import syn_speech_from_model_batch
from exp.exp_utils_bk import (copy_ref_speech, fine_adjust_configs,
                              get_synStyle_from_file, get_synText_from_file, renew_dict, combine_jsons)
from exp.vis_data_adaptor import convert_vis_psd_json, convert_attn_json, convert_attnEnh_json
from exp.extract_psd import extract_psdave, extract_psd_fine_class, extract_psd
from exp.visualization import vis_psd, vis_emo_crossAttn, vis_psd_enh, show_attn_map, show_two_attn_map
from exp.statcz_psd import statcz_psd_mcd, statcz_psd_fine_mcd

### TODO-S: combine model configs to one config.py
from config_dit_cross import VocosConfig
from config_dit_cross import ModelConfig, MelConfig
from config_mdit_cross import ModelConfig as mdit_ModelConfig
from config_dit_self import ModelConfig as self_Modelconfig

from config_dit_cross_cfm_distgl import ModelConfig as cfm_dit_Modelconfig
from config_dit_cross_cfm_distgl import ModelConfig as cfm_mdit_Modelconfig
from config_dit_self_cfm import ModelConfig as cfm_self_Modelconfig


from vis_data_adaptor import adapt_mix_2d_r_m1_m2, adapt_attn_2d_block_model
from vis2 import cst_attn_2d_block_model, cst_melattn_2d_type_model


## CONSTANTS
mname_config_ckpg_dict = {
    "cfm_dit_self": (cfm_self_Modelconfig, 60),
    "cfm_dit_cross_distgl": (cfm_dit_Modelconfig, 60),
    "cfm_mdit_cross_distgl": (cfm_mdit_Modelconfig, 300),
    "styletts2": ("", 300),
    "cfm_dit_cross_distgl_adaln6": (cfm_dit_Modelconfig, 60),
    "cfm_dit_cross_distgl_adaln6_oriResidual": (cfm_dit_Modelconfig, 60),  # add temp model
    "cfm_dit_cross_distgl_book5": (cfm_dit_Modelconfig, 60)
}

orderd_cmp_modelnames = ["reference", "styletts2", "cfm_mdit_cross_distgl"]
mel_config = MelConfig
voc_config= VocosConfig
vocoder_pt = '/home/rosen/Project/StableTTS/vocoders/vocos/checkpoints/generator_60.pt'

chk_pt = lambda model, epoch:f'../{model}/ckpt/checkpoint_{epoch}.pt'

def main(
        styles: List[Tuple[Any, Any, Any, Any, Any, Any]],
        synTexts: List[str],
        cmp_modelnames: ("", ""),
        out_dir: str = "option",  # middle result
        start_step=1,
        end_step=2,
        inference_config=None,
        vis_attn_config=None,
        vis_psd_config=None,
        vis_psdmel_config=None,
        save_attn_json_file=True
    ):
    """
    0: syn text by single moddel (preprare)
    1: syn fine-grained text by single moddel (preprare)
    2: vis attention compare
    3: vis psd contour compare
    4: psd diff statistics compare
    """
    ####### 0: syn text by single moddel (preprare) #######
    ckpt_func = lambda model, epoch: f'../ckpt/{model}/checkpoint_{epoch}.pt'


    if start_step <= 0 <= end_step:
        for model_name in cmp_modelnames:
            # OUT: out_dir/model_name/style_text_name/...
            style_syntex_name = "random"
            model_config, chk_pt = mname_config_ckpg_dict[model_name]
            model1_n_tts_voc = model_name, model_config, ckpt_func(model_name, chk_pt), mel_config, voc_config, vocoder_pt   # model_name, model_config,
            syn_speech_from_model_batch(model1_n_tts_voc, styles, synTexts, style_syntex_name, out_dir, inference_config=inference_config, save_attn_json_file=save_attn_json_file, save_attn=True)
    else:
        print("1. Skip speech synthesis!")

    ####### 1: extract psd #######
    if start_step <= 1 <= end_step:
        PSD_LEVEL = "phoneme"
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        prosody_dict = {}
        if PSD_LEVEL == "frame":
            psd_json_path = os.path.join(out_dir, "psd_{}.json".format("_".join(cmp_modelnames)))
            for model_name in cmp_modelnames:
                out_speech_dir = os.path.join(out_dir, model_name, "random")
                prosody_dict = extract_psd(mel_config, out_speech_dir, model_n=model_name, save_psd_file="", prosody_dict=prosody_dict)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
        elif PSD_LEVEL == "phoneme":
            psd_json_path = os.path.join(out_dir, "psdave_{}.json".format("_".join(cmp_modelnames)))
            prosody_psdave_dict = extract_psdave(mel_config, cmp_modelnames, out_dir)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_psdave_dict, sort_keys=True, indent=4))

    ####### 2. Vis attention map given attn file #######
    if start_step <= 2 <= end_step and len(cmp_modelnames) > 1:
        # combine attn_json
        attn_model_paths = [os.path.join(out_dir, "attn_{}_random.json".format(model_name)) for model_name in cmp_modelnames]
        cmp_modelnames_combine = "_".join(cmp_modelnames)
        attn_cmp_path = os.path.join(out_dir, f"attn_{cmp_modelnames_combine}.json")
        if len(attn_model_paths) == 2:
            combine_jsons(attn_model_paths[0], attn_model_paths[1], attn_cmp_path)
        else:
            IOError("more than 3 combination is not supported!")
        with open(attn_cmp_path, "r") as f:
            attn_dict = json.load(f)
        t, h, txt, emo, tg = vis_attn_config["show_t"], vis_attn_config["show_h"], vis_attn_config["show_txt"], vis_attn_config["show_emo"], vis_attn_config["tick_gran"]
        show_spk = vis_attn_config["show_spk"]
        # vis attn
        adpt_attn_data, png_n, title = adapt_attn_2d_block_model(attn_dict["spk" + show_spk], out_dir, show_t=t, show_h=h, show_txt=txt, show_emo=emo, tick_gran=tg)  # phoneme
        cst_attn_2d_block_model(adpt_attn_data, png_n, title)
        # vis attn&mel
        adpt_attn_mel_data, png_n, title = adapt_mix_2d_r_m1_m2(attn_dict["spk" + show_spk], out_dir, show_t=t, show_h=h, show_txt=txt, show_emo=emo, tick_gran=tg)  # syllable
        cst_melattn_2d_type_model(adpt_attn_mel_data, png_n, title)

    ####### 3. Vis psd contour given json file  #######
    # save pitch_dict_for_vis
    if start_step <= 3 <= end_step and len(cmp_modelnames) > 1:
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        cmp_modelnames_combine = "_".join(cmp_modelnames)
        psd_json_path = os.path.join(out_dir, "psdave_{}.json".format("_".join(cmp_modelnames)))
        #psd_json_path = "/hdd/StableTTS/exp/cfm_dstgl/psdave_cfm_dit_cross_distgl_cfm_mdit_cross_distgl_styletts2_reference.json"

        with open(psd_json_path, "r") as f:
            psd_cmp_dict = json.load(f)
        show_spk, show_text = vis_psd_config["show_spk"], vis_psd_config["show_txt"]
        psd_compare_png = os.path.join(out_dir, f"psd_{cmp_modelnames_combine}_ref{show_text[0]}_syn{show_text[1]}.png")

        pitch_cmp_path, energy_cmp_path = os.path.join(out_dir, f"pitch_{cmp_modelnames_combine}.json"), os.path.join(out_dir, f"energy_{cmp_modelnames_combine}.json")
        pitch_dict_for_vis, energy_dict_for_vis = convert_vis_psd_json(psd_cmp_dict["spk" + show_spk], show_text, cutpad_reference=False, save_dict=(pitch_cmp_path, energy_cmp_path))  # for_vis: {"emo1": {"model1": list(psd_len)}}}
        vis_psd(pitch_dict_for_vis, energy_dict_for_vis, psd_compare_png, show_text, ordered_lengend=orderd_cmp_modelnames)

    ####### 4. Statistics psd and mc given json file  #######
    if start_step <= 4 <= end_step:
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        cmp_modelnames_combine = "_".join(cmp_modelnames)
        psd_json_path = os.path.join(out_dir, "psd_{}.json".format(cmp_modelnames_combine))
        psd_statics_json = os.path.join(out_dir, f"stats_psd_{cmp_modelnames_combine}.json")
        with open(psd_json_path, "r") as f:
            prosody_dict = json.load(f)
        psd_mcd_stat_res = statcz_psd_mcd(prosody_dict, exclude_zero=True)  # {"spk": {"ang": {"modelA": [p1, e1, m1]}, "happy":{ "modelB": []]}}}
        with open(psd_statics_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mcd_stat_res, sort_keys=True, indent=4))
        print(psd_mcd_stat_res)
        # {"spk": {"ang": {"modelA": {"04": [p1, e1, m1]}}}}   <-02 06 10 14 18 22
        # {"spk": {"ang": {"modelA": {"spos": [p1, e1, m1]}}   <-spos, dpos


def main_fine(
        styles: List[Tuple[Any, Any, Any, Any, Any, Any]],
        synTexts: List[str],
        cmp_modelnames: ("", ""),
        fine_grained_name: (""),
        out_dir: str = "",  # middle result
        inference_config=None,
        need_syn_fine=False,
        need_psd_fine=False
):
    ####### 1: syn fine-grained text by single moddel (preprare) #######
    if need_syn_fine:
        for model_name in cmp_modelnames:
            for syn_text_fine, fine_name in (synTexts, fine_grained_name):
                out_speech_fine_dir = os.path.join(out_dir, model_name, fine_name)
                main_single_model((model_name, model_config_pt1, voc_config_pt), styles, syn_text_fine, out_speech_fine_dir, inference_config=inference_config, start_step=3, end_step=3)
    ####### 2: extract fine-grained psd #######
    if need_psd_fine:
        for model_name in cmp_modelnames:
            prosody_fine_dict = {}
            psd_model_path = os.path.join(out_dir, "psd_{}_{}.json".format(model_name, fine_grained_name[0]+"etc"))
            for syn_text_fine, fine_name in (synTexts, fine_grained_name):
                out_speech_fine_dir = os.path.join("", model_name, fine_name)
                prosody_dict = extract_psd(mel_config, out_speech_fine_dir, save_psd_file="")  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
                prosody_fine_dict.update(prosody_dict)
                #prosody_psdave_dict = extract_psdave(mel_config, in_speech_dir, save_psd_file="", average_phoneme=True, save_npy=False)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_model_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
    else:
        print("2. Skip fine-grained prosody extraction!")


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

if __name__ == "__main__":
    seed = 0
    import torch
    import argparse
    torch.manual_seed(seed)

    # Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name1", type=str, default="cfm_dit_cross_esdlj")
    parser.add_argument("--model_name2", type=str, default="cfm_mdit_cross")
    parser.add_argument("--epoch1", type=str, default="300")  # 510
    parser.add_argument("--epoch2", type=str, default="300")  # _617: 660, _618: 450
    #parser.add_argument("--style", type=str, default="data/esd_spk_txt_sort.txt")  # evalstyle
    #parser.add_argument("--txt", type=str, default="data/esd_unpara.txt")  # evaltxt_para.txt
    parser.add_argument("--style", type=str, default="data2/r1_test.txt")  # evalstyle
    #parser.add_argument("--txt", type=str, default="data2/s1.txt")  # evaltxt_para.txt
    parser.add_argument("--txt", type=str, default="data2/r1_in_txt_test.txt")  # evaltxt_para.txt

    args = parser.parse_args()

    # INPUT -> syn_styles (emo, spk, wav_p, psd_code), synTexts, and vis related ()
    #TEST_PART_NUM = 2
    syn_styles = get_synStyle_from_file(args.style, split_char='|', melstyle_type="codec")  # emotion changed
    synTexts = get_synText_from_file(args.txt)

    ## Vis related
    show_spk = "0013"  # for vis attention and psd contour   (Chimpion SPK18/TXT2, SPK17/TXT2)
    show_emo = "Surprise"  # Only in attention map
    show_text = 1
    #show_spks, show_emos, show_texts = [("0019", "Surprise", 0), ("0013", "Angry", 0), ("0017", "Surprise", 3)]   # spk0019_Surprise_ref3_syn3
    # spk0019_Surprise_ref3: he was still in the forest!
    vis_attn_config = {"show_spk": "0019", "show_t": 0, "show_h": 0, "show_txt": 3, "show_emo": "Surprise", "tick_gran": "syllable"}
    vis_psd_config = {"show_spk": "0013", "show_emo": "Surprise", "tick_gran": "syllable", "show_txt": (3, 4)}  # ref3:  15 - 19
    vis_psdmel_config = {"show_spk": "0013", "show_emo": "Surprise", "tick_gran": "syllable"}

    ### CHIMPION: he was still in the forest! (high pitch) <- spk19. txt0
    """
    0: Synthesize speech and generate attn_json
    1: extract psd_json
    2: vis attention
    3: vis psd
    4: compute statistics
    """
    # INPUT
    eval_models = ["cfm_dit_self", "cfm_dit_cross_distgl", "cfm_mdit_cross_distgl", "styletts2"]  # ["cfm_dit_cross_dstgl"], ["cfm_dit_cross_dstgl", "cfm_dit_cross_dstgl"]
    #eval_models = ["cfm_mdit_cross_distgl"]  # ["cfm_dit_cross_dstgl"], ["cfm_dit_cross_dstgl", "cfm_dit_cross_dstgl"]
    eval_models = ["cfm_mdit_cross_distgl", "cfm_dit_cross_distgl"]
    eval_models = ["cfm_dit_cross_distgl_book5"]  # cfm_dit_cross_distgl_adaln6_oriResidual, cfm_dit_cross_distgl_adaln6, cfm_dit_cross_distgl_adaln6_book5

    # OUTPUT
    out_dir = "/hdd/StableTTS/exp/cfm_dit_cross_distgl_test"
    inference_config = {"language": "english", "step": 50, "solver": 'dopri5', "cfg": 1}
    main(
        styles=syn_styles,
        synTexts=synTexts,
        cmp_modelnames=eval_models,
        out_dir=out_dir,
        start_step=0,
        end_step=0,
        vis_attn_config=vis_attn_config,
        vis_psd_config=vis_psd_config,
        vis_psdmel_config=vis_psdmel_config,
        inference_config=inference_config,
        save_attn_json_file=False
    )