import copy
from exp.exp_utils import (copy_ref_speech, fine_adjust_configs, get_synStyle_from_file, get_synText_from_file, renew_dict, combine_jsons, combine_two_jsons)
from tqdm import tqdm
import shutil
import sys, os, yaml, json
from inference_1_or_2 import syn_speech_by_second_model, get_second_model
from inference_decoTTS import syn_speech_by_second_model as syn_speech_by_second_model_deco, get_second_model as get_second_model_deco
from inference_origin import syn_speech, get_styletts2_model


mname_config = {
    "mdit_librittsesd_cutdurspn_pe": "drawspeech/config/mdit_librittsesd_cutdurspn_pe_infer.yaml",
    "monoDiT": "",
    "styletts2": ""
}

monoDiT_inference_args = {
    "alpha":"alpha",
    "beta":"beta",
    "diffusion_steps":10,
    "embedding_scale":1,
    "style_dim":"style_dim",
    "mix_ref_pe_type":"mix_ref_pe_type",
    "cfg_strength":"cfg_strength"
}


def inference_model_base(model, model_params, synTexts, ref_speechs, inference_args, output_dir):
    pass

def inference_decoDiT(ckpt, model_params, synTexts, ref_speechs, output_dir, inference_args, mix_ref_pe_types=None,
                      mono_guide_deltas=None, save_attn=False, save_cond=False, fuse_strength_gammas=None):

    get_second_model_deco()
    return attn_json_sub, psdcond_json_sub

def inference_monoDiT(ckpt, model_params, synTexts, ref_speechs, output_dir, inference_args, mix_ref_pe_types=None,
                      mono_guide_deltas=None, save_attn=False, save_cond=False, fuse_strength_gammas=None):
    """
    inference monoDiT given synTexts, ref_speechs and set of hypers (option)
    return:
        attn_json:
        psdcond_json:
    """
    # Get syn model/sampler
    second_model, sampler, model_params = get_second_model(ckpt=ckpt, config_f=model_params, model_name="mdit_cfm")

    # Synthesize by different hypers (2: ref_pe_type, monoDelta)
    attn_json = {}
    psdcond_json = {}
    model_infer_config_json = {}
    model_infer_config_f = os.path.join(output_dir, "model_infer_config.json")  # memo
    if mix_ref_pe_types is not None or mono_guide_deltas is not None or fuse_strength_gammas is not None:  # if given set of hypers
        for mix_ref_pe_type in mix_ref_pe_types:  # ["ref_pe", "none", "ref_pred_add"]
            for mono_guide_delta in mono_guide_deltas:
                if mix_ref_pe_type != "ref_pred_add":  # When not ref_pred_add, only use first fuse_strength (actually not used at all.)
                    fuse_strength_gammas_temp = [fuse_strength_gammas[0]]
                else:
                    fuse_strength_gammas_temp = fuse_strength_gammas
                for fuse_strength_gamma in fuse_strength_gammas_temp:

                    output_dir_abl_name = mix_ref_pe_type + "_m" + str(mono_guide_delta).replace(".", "").replace("-", "m") + \
                                     "_f" + str(fuse_strength_gamma).replace(".", "")
                    output_dir_abl = os.path.join(output_dir, output_dir_abl_name)
                    # three hyper for evaluation
                    inference_args["mix_ref_pe_type"] = mix_ref_pe_type
                    inference_args["mono_guide_delta"] = mono_guide_delta
                    inference_args["fuse_beta"] = fuse_strength_gamma

                    # save inference config
                    model_infer_config_json[output_dir_abl_name] = copy.deepcopy(inference_args)

                    attn_json_sub, psdcond_json_sub = syn_speech_by_second_model(synTexts, ref_speechs, output_dir_abl, second_model, sampler, model_params,
                                                               **inference_args, save_attn=save_attn, save_cond=save_cond, model_name=output_dir_abl_name)
                    # combine att_json_sub
                    if len(attn_json) == 0:
                        attn_json = attn_json_sub.copy()
                    else:
                        attn_current_json = attn_json.copy()
                        attn_json = combine_two_jsons(attn_current_json, attn_json_sub)

                    # combine psdcond_json_sub
                    if len(psdcond_json) == 0:
                        psdcond_json = psdcond_json_sub.copy()
                    else:
                        attn_current_json = psdcond_json.copy()
                        psdcond_json = combine_two_jsons(attn_current_json, psdcond_json_sub)
        with open(model_infer_config_f, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_infer_config_json, sort_keys=True, indent=4))
    else:
        attn_json, psdcond_json = syn_speech_by_second_model(synTexts, ref_speechs, output_dir, second_model, sampler, model_params,
                                   **inference_args, save_attn=save_attn, model_name="monoDiT")
    return attn_json, psdcond_json


def inference_styletts2(ckpt, model_params, synTexts, ref_speechs, output_dir, inference_args):
    model, sampler, model_params = get_styletts2_model(ckpt=ckpt, config_f=model_params, model_name="styletts2")
    syn_speech(synTexts, ref_speechs, output_dir, model, sampler, model_params, **inference_args)

def inference_styletts2_txt2mel():
    pass

def inference_hierspeech():
    pass


def inference_Dit(ckpt, model_params, synTexts, ref_speechs, output_dir, inference_args):
    # Get syn model/sampler
    second_model, sampler, model_params = get_second_model(ckpt=ckpt, config_f=model_params, model_name="mdit_cfm")
    attn_json, psdcond_json = syn_speech_by_second_model(synTexts, ref_speechs, output_dir, second_model, sampler, model_params,
                                           **inference_args, save_attn=False)


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
