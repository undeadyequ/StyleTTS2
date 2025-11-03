import shutil
import sys, os, yaml, json
sys.path.append('/home/rosen/Project/DrawSpeech_PyTorch')
os.chdir('/home/rosen/Project/DrawSpeech_PyTorch')
import torch
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Literal
from exp.exp_utils_bk import (copy_ref_speech, fine_adjust_configs,
                              get_synStyle_from_file, get_synText_from_file, renew_dict, combine_jsons)
from exp.vis_data_adaptor import convert_vis_psd_json, convert_attn_json, convert_attnEnh_json
from exp.extract_psd import extract_psdave, extract_psd_fine_class, extract_psd
from exp.visualization import vis_psd, vis_emo_crossAttn, vis_psd_enh, show_attn_map, show_two_attn_map, vis_mono_guide_mask, vis_dual_utmos_rmse
from exp.statcz_psd import statcz_psd_mcd, statcz_psd_fine_mcd

from vis_data_adaptor import adapt_mix_2d_r_m1_m2, adapt_attn_2d_block_model
from vis2 import cst_attn_2d_block_model, cst_melattn_2d_type_model
from drawspeech.infer_exp import infer_exp
from exp.mel_config import MelConfig          ################ BE CAREFUL; Must same with draw?.yaml #########
from tqdm import tqdm
from exp.exp_wer import evaluate_wer
from drawspeech.utilities.guide_mask import make_guided_attention_masks2
from drawspeech.utilities.vis import save_plot

def get_infer_json(styles, synTexts, ref_json, gd_dir=None):
    """
    Get infer json (for training), and copy ground truth speech (for WER)
    """
    # create name_phoneme_dict
    data = json.load(open(ref_json, "r"))["data"]
    name_phoneme_dict = {}  # searching phoneme by name (for reference speech path)
    text_phoneme_dict = {}  # searching phoneme by text (for synText)
    for d in tqdm(data):
        text = d["transcription"]
        basename = d["wav"].split("/")[-1].replace(".wav", "")
        name_phoneme_dict[basename] = (d["phonemes"], d["duration"])
        text_phoneme_dict[text] = (d["phonemes"], d["wav"], basename)

    # make infer_json
    ref_texts = []
    data = []
    for j, style in enumerate(styles):
        spk, emo, ref_txt, speech_path = style
        ref_texts.append(ref_txt) if ref_txt not in ref_texts else ref_texts
        r_id = ref_texts.index(ref_txt)

        basename = speech_path.split("/")[-1].replace(".wav", "")
        for k, syntext in enumerate(synTexts):
            if j == 0:
                # copy ground truth speech given synText (for WER experiment)
                gd_wav_path = os.path.join(dataset_rootdir, text_phoneme_dict[syntext][1])
                dst_path = os.path.join(gd_dir, text_phoneme_dict[syntext][-1] + ".wav")
                shutil.copy(gd_wav_path, dst_path)
                txt_f = f'{gd_dir}/{text_phoneme_dict[syntext][-1]}.lab'
                with open(txt_f, "w") as file1:
                    file1.write(syntext)


            data.append(
                {
                    "wav": speech_path,           #  ref
                    "transcription": syntext,     # syntext
                    "ref_transcription": ref_txt,
                    "ref_phonemes": name_phoneme_dict[basename][0],
                    "phonemes": text_phoneme_dict[syntext][0],  # SynText, used for phoneme embedding
                    "duration": name_phoneme_dict[basename][1],  #  ref
                    "emo": emo, #  ref
                    "spk": spk, #  ref
                    "r_id": r_id,  #  ref
                    "t_id": k
                })
    return {"data": data}


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

    if not os.path.isfile(infer_json_path):
        infer_json = get_infer_json(syn_styles, synTexts, ref_json, gd_dir=gd_speech_dir)  # write ref/syn phoneme in infer.json
        json.dump(infer_json, open(infer_json_path, "w"), indent=1, ensure_ascii=False)
    else:
        infer_json = json.load(open(infer_json_path, "r"))

    ####### 0: syn text by single moddel (preprare) #######
    if start_step <= 0 <= end_step:
        for model_name in cmp_modelnames:
            if model_name in OTHER_CMP_MODELS:
                continue
            config_yaml = mname_config[model_name]
            ## exp_group_name, exp_name is used to locate ckpt
            exp_name = os.path.basename(config_yaml.split(".")[0])
            if exp_name.endswith("_infer"):
                exp_name = exp_name[:-6]
            exp_group_name = os.path.basename(os.path.dirname(config_yaml))
            config_yaml_path = os.path.join(config_yaml)
            config_yaml_dict = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

            ## Output
            out_speech_dir = os.path.join(out_dir, model_name, style_syntex_name)  #out_dir/model_name/[random/CondA/CondB]]/...
            out_attn_dir = os.path.join(out_dir, model_name, style_syntex_name + "_attn")  # few attn is enough?
            ref_speech_dir = os.path.join(out_dir, "reference", style_syntex_name)  #
            attn_json_path = os.path.join(out_dir, "attn_{}_{}.json".  # {spk/emo/model: {speech_id:[], syn_phonemes:[], ref_phonemes:[], q_dur:[], k_dur:[]}}
                                          format(model_name, style_syntex_name))  # only needed in random synthesis
            if not os.path.isdir(out_speech_dir):
                Path(out_speech_dir).mkdir(exist_ok=True, parents=True)
            if not os.path.isdir(ref_speech_dir):
                Path(ref_speech_dir).mkdir(exist_ok=True, parents=True)
            if not os.path.isdir(out_attn_dir) and save_attn and "dit" in model_name:
                Path(out_attn_dir).mkdir(exist_ok=True, parents=True)

            if "dit" not in model_name:
                save_attn = False
            # syn speech, save attn, copy ref, create attn_json (for exp)
            _, attn_dict = infer_exp(infer_json, config_yaml_dict, config_yaml_path, exp_group_name, exp_name, syn_styles, synTexts,
                                  out_speech_dir, ref_speech_dir, out_attn_dir, batch_size=8, save_attn=save_attn)  # should output out_attn_dir and attn_json_path
            if save_attn_json_file and attn_dict is not None:
                with open(attn_json_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(attn_dict, sort_keys=True, indent=4))
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
                out_speech_dir = os.path.join(out_dir, model_name, "random")
                prosody_dict = extract_psd(mel_config, out_speech_dir, model_n=model_name, save_psd_file="", prosody_dict=prosody_dict)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
        elif psd_level == "phoneme":
            psd_json_path = os.path.join(out_dir, "psdave_{}.json".format("_".join(cmp_modelnames)))
            prosody_psdave_dict = extract_psdave(mel_config, cmp_modelnames, out_dir)  # {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}}
            with open(psd_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(prosody_psdave_dict, sort_keys=True, indent=4))

    ####### 2. Statistics psd and mc given psd_{}.json (output of 1) #######
    if start_step <= 2 <= end_step:
        cmp_modelnames.append("reference") if "reference" not in cmp_modelnames else None
        cmp_modelnames_combine = "_".join(cmp_modelnames)
        if psd_level == "frame":
            psd_json_path = os.path.join(out_dir, "psd_{}.json".format(cmp_modelnames_combine))
            psd_statics_json = os.path.join(out_dir, f"stats_psd_{cmp_modelnames_combine}.json")
            psd_statics_mean_json = os.path.join(out_dir, f"stats_psd_{cmp_modelnames_combine}_mean.json")
        elif psd_level == "phoneme":
            psd_json_path = os.path.join(out_dir, "psdave_{}.json".format(cmp_modelnames_combine))
            psd_statics_json = os.path.join(out_dir, f"statsave_psd_{cmp_modelnames_combine}.json")
            psd_statics_mean_json = os.path.join(out_dir, f"statsave_psd_{cmp_modelnames_combine}_mean.json")

        with open(psd_json_path, "r") as f:
            prosody_dict = json.load(f)
        psd_mcd_stat_res, psd_mean_stat_res = statcz_psd_mcd(prosody_dict, exclude_zero=True)  # {"spk": {"ang": {"modelA": [p1, e1, m1]}, "happy":{ "modelB": []]}}}
        with open(psd_statics_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mcd_stat_res, sort_keys=True, indent=4))
        with open(psd_statics_mean_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(psd_mean_stat_res, sort_keys=True, indent=4))
        print(psd_mean_stat_res)
        # {"spk": {"ang": {"modelA": {"04": [p1, e1, m1]}}}}   <-02 06 10 14 18 22
        # {"spk": {"ang": {"modelA": {"spos": [p1, e1, m1]}}   <-spos, dpos

    ####### 3: WER #######
    if start_step <= 3 <= end_step:
        for model_name in cmp_modelnames:
            out_speech_dir = os.path.join(out_dir, model_name, style_syntex_name)  #out_dir/model_name/[random/CondA/CondB]]/...
            evaluate_wer(out_speech_dir, output_csv=f"{out_dir}/wer_{model_name}.csv")
        # do wer on gd_speech
        if not os.path.isfile(f"{out_dir}/wer_gd_speech.csv"):
            evaluate_wer(os.path.join(out_dir, "gd_speech"), output_csv=f"{out_dir}/wer_gd_speech.csv")

    ####### 4. Vis psd contour given json file  #######
    img_out = os.path.join(out_dir, "img_out")
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
        psd_compare_png = os.path.join(img_out,
                                       f"psd_{cmp_modelnames_combine}_ref{show_text[0]}_syn{show_text[1]}.png")

        pitch_cmp_path, energy_cmp_path = os.path.join(out_dir,
                                                       f"vis_pitch_{cmp_modelnames_combine}.json"), os.path.join(
            out_dir, f"vis_energy_{cmp_modelnames_combine}.json")
        pitch_dict_for_vis, energy_dict_for_vis = convert_vis_psd_json(
            psd_cmp_dict["spk" + show_spk], show_text, cutpad_reference=False,
            save_dict=(pitch_cmp_path, energy_cmp_path))  # for_vis: {"emo1": {"model1": list(psd_len)}}}
        vis_psd(pitch_dict_for_vis, energy_dict_for_vis, psd_compare_png, show_text,
                ordered_lengend=orderd_cmp_modelnames)

    ######## 5. Vis attention/mel map
    if start_step <= 5 <= end_step:
        dul_axis_data = {"y1": [40, 50, 60], "y2": [3.01, 3.22, 3.21], "x": [0.2, 0.5, 0.8]}
        dual_rmse_utmos_png = os.path.join(img_out, f"vis_dual_rmse_utmos.pdf")
        vis_dual_utmos_rmse(dul_axis_data, out_pic=dual_rmse_utmos_png)

        guid_masks_png = os.path.join(img_out, f"vis_guid_masks.pdf")
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
                                                                 model_ab=(cmp_modelnames[0],
                                                                           cmp_modelnames[1]))  # phoneme
        cst_attn_2d_block_model(adpt_attn_data, png_n, title)
        # vis attn&mel
        adpt_attn_mel_data, png_n, title = adapt_mix_2d_r_m1_m2(attn_dict["spk" + show_spk], out_dir, show_t=t,
                                                                show_h=h, show_txt=txt, show_emo=emo, tick_gran=tg,
                                                                model_ab=(cmp_modelnames[0],
                                                                          cmp_modelnames[1]))  # syllable
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

if __name__ == "__main__":
    """
    -1: Generate infer.json (3Type) from given style and text file
    0: Synthesize speech dir for each model and generate attn_json
    1: extract psd_json
    2: compute statistics
    3: WER
    4: vis attn
    5: vis attn/mel
    On doing: extract SIM-O and SIM-R  (?)
    """
    seed = 0
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()

    ##### 1. CONSTANTS
    dataset_name_rootdir = {"esd": "/hdd/ESD", "libritts": "LibriTTS_16k"}
    meta_data_dir = "/home/rosen/Project/DrawSpeech_PyTorch/data/dataset/metadata"
    ref_json = {
        "esd": os.path.join(meta_data_dir, "esd/train.json"),
        "libritts": os.path.join(meta_data_dir, "libritts16k_cutdur/train.json")}
    #mname_config = lambda config_name: os.path.join("drawspeech/config", config_name)
    mname_config = {
        "mdit_librittsesd_cutdurspn_pe": "drawspeech/config/mdit_librittsesd_cutdurspn_pe_infer.yaml"
    }
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
        "libritts": {"show_spk": "0019", "show_emo": "Surprise", "tick_gran": "syllable", "show_txt": (3, 4)}
    }  # ref3:  15 - 19
    vis_psdmel_config = {
        "esd": {"show_spk": "0013", "show_emo": "Surprise", "tick_gran": "syllable"},
        "libritts": {"show_spk": "0013", "show_emo": "Surprise", "tick_gran": "syllable"}}

    parser.add_argument("--style", type=str, default="exp/data2/r1_50.txt")  # evalstyle  r1_test  r1 libri_r1, r1_v2
    parser.add_argument("--txt", type=str, default="exp/data2/s1_5.txt")  # evaltxt_para.txt libri_s1, s1_v2
    args = parser.parse_args()

    ######### 3. INPUT
    # orderd_cmp_modelnames = ["reference", "styletts2", "lddpm_dit_pe_libritts"]
    orderd_cmp_modelnames = ["reference", "styletts2", "mdit_librittsesd_cutdurspn_pe"]
    dataset_name = "esd"    # libritts esd
    dataset_rootdir = dataset_name_rootdir[dataset_name]   # LibriTTS_16k  ESD
    eval_models = ["styletts2", "mdit_librittsesd_cutdurspn_pe"]

    # INPUT -> syn_styles (emo, spk, wav_p, psd_code), synTexts, and vis related ()
    #TEST_PART_NUM = 2
    ### CHIMPION: he was still in the forest! (high pitch) <- spk19. txt0
    # eval_models = ["drawspeech_libritts_16k_spk_cutdur", "drawspeech_libritts_mdit_16k_cutdur_phase2"] # ["cfm_dit_self", "cfm_dit_cross_distgl", "cfm_mdit_cross_distgl", "styletts2"]

    ######### OUTPUT
    out_dir = f"/hdd/drawspeech/log/exp/lddpm_{dataset_name}_basic"   # esd
    syn_styles = get_synStyle_from_file(args.style, split_char='|', melstyle_type="codec", dataset_name=dataset_name)  # emotion changed
    synTexts = get_synText_from_file(args.txt)

    EVAL_RANDOM = True
    EVAL_FINE1 = False
    EVAL_FINE2 = False
    if EVAL_RANDOM:
        NUM = 1
        for i in range(NUM):
            for j in range(NUM):
                vis_psd_config[dataset_name]["show_txt"] = (i, j)
                main(
                    syn_styles=syn_styles,
                    synTexts=synTexts,
                    cmp_modelnames=eval_models,
                    ref_json=ref_json[dataset_name],
                    out_dir=out_dir,
                    start_step=0,
                    end_step=0,
                    mel_config=mel_config,
                    vis_attn_config=vis_attn_config[dataset_name],
                    vis_psd_config=vis_psd_config[dataset_name],
                    vis_psdmel_config=vis_psdmel_config[dataset_name],
                    save_attn_json_file=True,
                    style_syntex_name="random",
                    infer_json_name="infer.json",
                    psd_level="phoneme",
                    save_attn=False,
                )

    if EVAL_FINE1:
        out_dir = "/hdd/drawspeech/log/exp/lddpm_basic/fine_lenRate"  #
        style_syn_f_dir = "/home/rosen/Project/StableTTS/exp/data2"
        ref_style_f = os.path.join(style_syn_f_dir, "r2.txt")

        fine_categ_labels = ["05", "1", "2"]
        fine_categ_fs = [os.path.join(style_syn_f_dir, "s2_" + fine_cat + ".txt") for fine_cat in fine_categ_labels]
        syn_styles = get_synStyle_from_file(ref_style_f, split_char='|', melstyle_type="codec")  # emotion changed
        fine_synTexts = [get_synText_from_file(fine_categ_f) for fine_categ_f in fine_categ_fs]

        for fine_cate_label, fine_synText in zip(fine_categ_labels, fine_synTexts):
            main(
                syn_styles=syn_styles,
                synTexts=fine_synText,
                cmp_modelnames=eval_models,
                ref_json=ref_json[dataset_name],
                out_dir=out_dir,
                start_step=2,
                end_step=2,   # Only Step 0, 1, 4
                mel_config=mel_config,
                vis_attn_config=vis_attn_config[dataset_name],
                vis_psd_config=vis_psd_config[dataset_name],
                vis_psdmel_config=vis_psdmel_config[dataset_name],
                save_attn_json_file=True,
                style_syntex_name=fine_cate_label,
                infer_json_name=f"infer_{fine_cate_label}.json",
                psd_level="phoneme",
                save_attn=False)

    if EVAL_FINE2:
        out_dir = "/hdd/drawspeech/log/exp/lddpm_basic/fine_pos"  # Selected Syntext by parts of speech
        style_syn_f_dir = "/home/rosen/Project/StableTTS/exp/data2"
        ref_style_f = os.path.join(style_syn_f_dir, "r2.txt")

        fine_categ_labels = ["same_pos", "diff_pos"]
        fine_categ_fs = [os.path.join(style_syn_f_dir, "s3_" + fine_cat + ".txt") for fine_cat in fine_categ_labels]
        syn_styles = get_synStyle_from_file(ref_style_f, split_char='|', melstyle_type="codec")  # emotion changed
        fine_synTexts = [get_synText_from_file(fine_categ_f) for fine_categ_f in fine_categ_fs]

        for fine_cate_label, fine_synText in zip(fine_categ_labels, fine_synTexts):
            main(
                syn_styles=syn_styles,
                synTexts=fine_synText,
                cmp_modelnames=eval_models,
                ref_json=ref_json[dataset_name],
                out_dir=out_dir,
                start_step=0,
                end_step=1,  # Only Step 0, 1, 4
                mel_config=mel_config,
                vis_attn_config=vis_attn_config[dataset_name],
                vis_psd_config=vis_psd_config[dataset_name],
                vis_psdmel_config=vis_psdmel_config[dataset_name],
                save_attn_json_file=True,
                style_syntex_name=fine_cate_label,
                infer_json_name=f"infer_{fine_cate_label}.json",
                psd_level="phoneme",
                save_attn=False)