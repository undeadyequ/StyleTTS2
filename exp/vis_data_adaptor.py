"""
1. psd

2. attn (subplots of 1d, 2d)

3. melspectrogram

4. mixed of attn mel
"""

from pymcd.mcd import Calculate_MCD
mcd_toolbox = Calculate_MCD(MCD_mode="MCD-DTW")
import sys, os, yaml, json
import numpy as np
from exp.exp_utils import convert_xydur_xybox, clean_phone, cut_pad_a2b_len_left
from exp.syllable import extend_phone2syl


def convert_vis_psd_json(prosody_dict_json, show_ref_syn_id=(0, 0), cutpad_reference=False, fine_categ_labels=["random"], save_dict=("", "")):
    """
    prosody_dict_json: {"emo1": {"model1": {"psd1/phone": []}}} -> {"emo1": {"model1": (pitch_list, phone_list)}}}  {}
    """
    nonref_model_name = ""
    pitch_dict_for_vis = dict()  # for_vis: {"emo1": {"model1": list(p_len)}}}
    for emo, m_psd in prosody_dict_json.items():
        pitch_dict_for_vis[emo] = {}
        for model, psd in m_psd.items():
            if model not in pitch_dict_for_vis[emo].keys():
                if fine_categ_labels[0] in psd.keys():
                    psd = psd[fine_categ_labels[0]]
                if model == "reference":
                    real_text_id = show_ref_syn_id[0]
                else:
                    r_id, s_id = show_ref_syn_id
                    real_text_id = max([i if "ref{}_syn{}".format(r_id, s_id) in speechid else -1 for i, speechid in enumerate(psd["speechid"])])
                pitch_model = psd["pitch"][real_text_id]
                phonemes = psd["phonemes"][real_text_id]
                if model != "reference":
                    nonref_model_name = model   # appeae once
                if cutpad_reference and model == "reference":
                    pitch_nonref_model = m_psd[nonref_model_name][fine_categ_labels[0]]["pitch"][real_text_id]
                    phoneme_nonref_model = m_psd[nonref_model_name][fine_categ_labels[0]]["phonemes"][real_text_id]
                    pitch_model = cut_pad_a2b_len_left(np.array(pitch_model), len(pitch_nonref_model)).tolist()  # modified reference pitch to same length as model
                    phonemes = cut_pad_a2b_len_left(np.array(phonemes), len(phoneme_nonref_model)).tolist()  # modified reference phonemes to same length as model
                pitch_dict_for_vis[emo][model] = (pitch_model, phonemes)

    energy_dict_for_vis = dict()  # {"emo1": {"model1": list(p_len)}}}
    for emo, m_psd in prosody_dict_json.items():
        energy_dict_for_vis[emo] = {}
        for model, psd in m_psd.items():
            if model not in energy_dict_for_vis[emo].keys():
                if fine_categ_labels[0] in psd.keys():
                    psd = psd[fine_categ_labels[0]]
                #real_text_id = [i if "_".join(txt_name.split("_")[-2:]) == str(show_ref_syn_id) else -1 for i, txt_name in enumerate(psd["speechid"])][0]   # psd["pitch"] is not sorted
                if model == "reference":
                    real_text_id = show_ref_syn_id[0]
                else:
                    r_id, s_id = show_ref_syn_id
                    real_text_id = max([i if "ref{}_syn{}".format(r_id, s_id) in speechid else -1 for i, speechid in enumerate(psd["speechid"])])
                energy_model = psd["energy"][real_text_id]
                phonemes = psd["phonemes"][real_text_id]
                if cutpad_reference and model == "reference":
                    energy_nonref_model = m_psd[nonref_model_name][fine_categ_labels[0]]["energy"][real_text_id]
                    phoneme_nonref_model = m_psd[nonref_model_name][fine_categ_labels[0]]["phonemes"][real_text_id]
                    energy_model = cut_pad_a2b_len_left(np.array(energy_model), len(energy_nonref_model)).tolist()  # modified reference pitch to same length as model
                    phonemes = cut_pad_a2b_len_left(np.array(phonemes), len(phoneme_nonref_model)).tolist()  # modified reference phonemes to same length as model
                energy_dict_for_vis[emo][model] = (energy_model, phonemes)

    if len(save_dict[0]) != 0:
        vis_pitch_json, vis_energy_json = save_dict
        with open(vis_pitch_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(pitch_dict_for_vis, sort_keys=True, indent=4))
        with open(vis_energy_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(energy_dict_for_vis, sort_keys=True, indent=4))
    return pitch_dict_for_vis, energy_dict_for_vis


def modify_dur_phone(durs, phones, tick_gran="syllable", given_syl=[[], [], []]):
    """clean/modify phones to syllabels, with durs"""
    if len(durs) != len(phones):
        raise IOError("len of durs {} and phones {} should be same".format(len(durs), len(phones)))
    phones, durs = clean_phone(phones, durs, internum=0)  # clean "", intermark (11 or 0)
    print(phones)
    print(durs)
    if tick_gran == "syllable":
        syn_labels, syn_durs = extend_phone2syl(phones, durs, intermark="0")  # change phone to syllable
    else:
        syn_labels, syn_durs = clean_phone(phones, durs)  # clean "", 11
    syn_durs_inc, ref_durs_inc = [0], [0]
    syn_durs_inc.extend([sum(syn_durs[:i + 1]) for i in range(len(syn_durs))])
    syn_labels.append("")  # to align label to the left
    if len(syn_durs_inc) != len(syn_labels):
        raise IOError("durs and phones should have same lens {} {}".format(
            len(syn_durs_inc), len(syn_labels)))
    return syn_durs_inc, syn_labels


def phone2syl(syn_phones, syn_durs, syl_start_index):
    """
    gather phones to syllables, with durs
    :param syn_phones: list of phones
    :param syn_durs: list of durs
    :param syl_start_index: list of syllable start index
    :return:
    xticks:    ["a", ... ""]
    x_ticklabs:[0, ... sum(dur)]
    """
    if len(syn_phones) != len(syn_durs):
        raise IOError("len of syn_phones {} and syn_durs {} should be same".format(len(syn_phones), len(syn_durs)))
    syn_syls = []
    syn_syls_durs = []
    for i in range(len(syl_start_index) - 1):
        start = syl_start_index[i]
        end = syl_start_index[i + 1]
        syn_syls_durs.append(sum(syn_durs[start:end]))
        syn_syls.append("".join(syn_phones[start:end]))
    end = syl_start_index[-1]
    syn_syls_durs.append(sum(syn_durs[end:]))
    syn_syls.append("".join(syn_phones[end:]))

    syn_syls_durs_inc = [0]
    syn_syls_durs_inc.extend([sum(syn_syls_durs[:i + 1]) for i in range(len(syn_syls_durs))])
    syn_syls.append("")  # to align label to the left
    return syn_syls_durs_inc, syn_syls

def adapt_mix_2d_r_m1_m2(attn_dict_json, result_dir, show_t=0, show_b=5, show_h=0, show_txt=0, show_emo="Angry", tick_gran = "syllable",
                         model_ab=("cfm_dit_cross_distgl", "cfm_mdit_cross_distgl")):
    """
    2d attns with dim0=block, dim1=model
    content for each:
        title:
        data:
        xlabel, xticks, x_ticklabs:
        ylabel, yticks, y_ticklabs: (only for attention)
        kwargs: {
                xy_ticklabs_bold_index: ([], []),
                xy_auxline: ([], []),
                xy_rectangle: xywh_list
                xy_rectangle_bold_index: []}
    """
    batch_n = 0
    mix_2d_b_m_json = {
        "0_0": ["Reference"],     #
        "0_1": ["DiT"],  # Attn: title, attn, x_ltl_set, y_ltl_set, kwargs
        "0_2": ["mDiT"],
        "1_0": [""],     # blank
        "1_1": ["DiT"],  # Mel:  title, speech, x_ltl_set, kwargs
        "1_2": ["mDiT"],
    }
    attn_kwargs = {"xy_ticklabs_bold_index": ([4], [4]), "xy_auxline": (None, None), "xy_rectangle": None, "xy_rectangle_bold_index": [2],
        "fontsize": 12, "x_rotation": 45, "y_rotation": 45, "yticklabel_rotation": 90}
    speech_kwargs = {"xy_ticklabs_bold_index": ([4], [4]), "xy_auxline": (None, None), "xy_rectangle": None, "xy_rectangle_bold_index": [2],
                "fontsize": 12, "x_rotation": 45, "y_rotation": 45}

    #col2model = {"0": "reference", "1": "ddpm_dit_cross", "2": "ddpm_mdit_cross"}
    col2model = {"0": "reference", "1": model_ab[0], "2": model_ab[1]}

    syn_clean_syl_start_index = [0, 4, 7, 10, 15, 18]
    ref_clean_syl_start_index = [0, 3, 7, 12, 15, 19, 21]

    for k, v in mix_2d_b_m_json.items():
        datatype_index, model_index = k.split("_")
        model = col2model[model_index]
        if k in ["0_1", "0_2"]:   # attention
            crossAttn_f = os.path.join(result_dir, model, "random_attn/", attn_dict_json[show_emo][model]["speechid"][show_txt] + ".npy")
            attn = np.load(crossAttn_f, allow_pickle=True)
            if len(attn.shape) == 6:
                attn = attn[show_t, show_b, batch_n, show_h, ...]  # [time, block_n, batch, head, t_t, t_s]
            elif len(attn.shape) == 5:
                attn = attn[show_t, show_b, show_h, ...]  #  [time, block_n, head, t_t, t_s] # No batch
            else:
                attn = attn[show_b, show_h, ...]  # [block_n, head, t_t, t_s]    # in synthesize_from_batch, batch dim is removed

            attn = np.transpose(attn)
            syn_phones = attn_dict_json[show_emo][model]["syn_phonemes"][show_txt]
            syn_durs = attn_dict_json[show_emo][model]["q_dur"][show_txt]
            ref_phones = attn_dict_json[show_emo][model]["ref_phonemes"][show_txt]
            ref_durs = attn_dict_json[show_emo][model]["k_dur"][show_txt]
            syn_phones, syn_durs = clean_phone(syn_phones, syn_durs, internum=0)  # clean "", intermark (11 or 0)
            ref_phones, ref_durs = clean_phone(ref_phones, ref_durs, internum=0)  # clean "", intermark (11 or 0)
            xticks, x_ticklabs = phone2syl(syn_phones, syn_durs, syl_start_index=syn_clean_syl_start_index)
            yticks, y_ticklabs = phone2syl(ref_phones, ref_durs, syl_start_index=ref_clean_syl_start_index)
            """
            xticks, x_ticklabs = modify_dur_phone(attn_dict_json[show_emo][model]["k_dur"][show_txt],
                                                  attn_dict_json[show_emo][model]["ref_phonemes"][show_txt],
                                                  tick_gran=tick_gran)
            yticks, y_ticklabs = modify_dur_phone(attn_dict_json[show_emo][model]["q_dur"][show_txt],
                                                  attn_dict_json[show_emo][model]["syn_phonemes"][show_txt],
                                                  tick_gran=tick_gran)  ###### Need change for Upara ataks #######
            """
            mix_2d_b_m_json[k].extend([attn, ("", xticks, x_ticklabs), ("", yticks, y_ticklabs), attn_kwargs])
        elif k in ["0_0", "1_1", "1_2"]:                            # mel
            speech_dir = os.path.join(result_dir, model, "random")
            if model == "reference":   # no reference in attn_json
                model = col2model["1"]
                syn_speech = attn_dict_json[show_emo][col2model["1"]]["speechid"][show_txt]
                ref_speech = "_".join(syn_speech.split("_")[:-1])
                speech = os.path.join(speech_dir, ref_speech + ".wav")
            else:
                speech = os.path.join(speech_dir, attn_dict_json[show_emo][col2model["1"]]["speechid"][show_txt] + ".wav")

            xticks, x_ticklabs = modify_dur_phone(attn_dict_json[show_emo][model]["q_dur"][show_txt],
                                                  attn_dict_json[show_emo][model]["syn_phonemes"][show_txt],
                                                  tick_gran=tick_gran)
            mix_2d_b_m_json[k].extend([
                speech,
                ("", xticks, x_ticklabs),
                speech_kwargs])

    png_n = "attn_mel_of_block{}_model{}_emo{}_text{}.png".format(show_b, "dit_mdit", show_emo, show_txt)
    png_n = os.path.join(result_dir, png_n)
    title = "Relationship between attention monotonicity and sequential style alignment for DiT and mDiT blocks"
    return mix_2d_b_m_json, png_n, title

def adapt_attn_2d_block_model(attn_dict_json, result_dir, show_t=0, show_h=0, show_txt=0, show_emo="Angry", tick_gran="phoneme",
                              model_ab=("cfm_dit_cross_distgl", "cfm_mdit_cross_distgl")):
    """
    args:
    attn_dict_json: {emo: model: }

    out: attn_2d_b_m_json:
        X_Y: [attn,(x_label, xticks, x_ticklabs), (y_label, yticks, y_ticklabs), kwargs])

    content for each:
        title:
        attn_matrix:
        xlabel, xticks, x_ticklabs:
        ylabel, yticks, y_ticklabs: (attn.shape[1] == len(x_lab) == len(xticks))
        kwargs: {
                xy_ticklabs_bold_index: ([], []),
                xy_auxline: ([], []),
                xy_rectangle: xywh_list
                xy_rectangle_bold_index: []}
    """
    row2block = {"0": 0, "1": 1, "2": 2, "3": 3}
    batch_n = 0
    attn_2d_b_m_json = {
        "0_0": ["DiT"],  # block = 0, model = 0
        "0_1": ["mDiT"],
        "1_0": [""],
        "1_1": [""],
        "2_0": [""],
        "2_1": [""],
        "3_0": [""],
        "3_1": [""],
    }
    syn_clean_syl_start_index = [0, 4, 7, 10, 15, 18]           # Manual set
    ref_clean_syl_start_index = [0, 3, 7, 12, 15, 19, 21]

    for k, v in attn_2d_b_m_json.items():
        b, m = k.split("_")
        ### Variable: model and block
        model = model_ab[0] if m == "0" else model_ab[1]
        block = row2block[b]

        crossAttn_dir = os.path.join(result_dir, model,  "random_attn/")
        crossAttn_f = crossAttn_dir + attn_dict_json[show_emo][model]["speechid"][show_txt] + ".npy"
        attn = np.load(crossAttn_f, allow_pickle=True) # [time, block_n, batch, head, t_t, t_s]
        if len(attn.shape) == 6:
            attn = attn[show_t, block, batch_n, show_h, ...]  # [time, block_n, batch, head, t_t, t_s]
        elif len(attn.shape) == 5:
            attn = attn[block, batch_n, show_h, ...]  # [block_n, batch, head, t_t, t_s]   #
        else:
            attn = attn[block, show_h, ...]  # [block_n, head, t_t, t_s]    # in synthesize_from_batch, batch dim is removed
        attn = np.transpose(attn)
        syn_phones = attn_dict_json[show_emo][model]["syn_phonemes"][show_txt]
        syn_durs = attn_dict_json[show_emo][model]["q_dur"][show_txt]
        ref_phones = attn_dict_json[show_emo][model]["ref_phonemes"][show_txt]
        ref_durs = attn_dict_json[show_emo][model]["k_dur"][show_txt]
        #xticks, x_ticklabs = modify_dur_phone(syn_phones, syn_durs, tick_gran=tick_gran)
        #yticks, y_ticklabs = modify_dur_phone(ref_phones, ref_durs, tick_gran=tick_gran)

        syn_phones, syn_durs = clean_phone(syn_phones, syn_durs, internum=0)  # clean "", intermark (11 or 0)
        ref_phones, ref_durs = clean_phone(ref_phones, ref_durs, internum=0)  # clean "", intermark (11 or 0)
        xticks, x_ticklabs = phone2syl(syn_phones, syn_durs, syl_start_index=syn_clean_syl_start_index)
        yticks, y_ticklabs = phone2syl(ref_phones, ref_durs, syl_start_index=ref_clean_syl_start_index)
        #xywh_list = convert_xydur_xybox(xticks, yticks)
        kwargs = {
            "xy_ticklabs_bold_index": (None, None),
            "xy_auxline": (None, None),
            "xy_rectangle": None,  # xywh_list
            "xy_rectangle_bold_index": [2],
            "fontsize": 12,
            "x_rotation": 45,
            "y_rotation": 45,
            "yticklabel_rotation": 90
        }
        y_label = None
        x_label = "Synthesize phonemes"
        if int(b) < 3:                        # (0/1/2, 0)
            x_ticklabs = None
            x_label = ""
        if int(m) == 1:                       # (, 1)
            y_ticklabs = None
            y_label = "Block={}".format(row2block[b])
            kwargs["yticklabel_rotation"] = 0
        if int(m) == 0 and int(b) == 2:       # (2,0)
            y_label = "Reference phonemes"

        attn_2d_b_m_json[k].extend([
            attn,(x_label, xticks, x_ticklabs),
            (y_label, yticks, y_ticklabs),
            kwargs])
    png_n = "attn_of_block{}_model{}_emo{}_text{}.png".format("012", "dit_mdit", show_emo, show_txt)
    png_n = os.path.join(result_dir, png_n)
    title = "Cross-attention score map of the first (1st, 2nd) and the last 2 layers (5st, 6st) of the DiT and mDiT blocks (number of layers is 6)"
    return attn_2d_b_m_json, png_n, title

def convert_attn_json(attn_dict_json, attn_dir, show_t=0, show_b=0, show_h=0, show_txt=0):
    """
    For visualizing crossAttn
    - auxiliary line given mfa duration ?

    """
    batch_n = 0
    crossAttn_dict_for_vis = dict()  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}  <- choose t and h from attn_map_dict
    for emo, m_psd in attn_dict_json.items():
        for model, psd in m_psd.items():
            if model != "reference":
                crossAttn_dir = os.path.join(attn_dir, model + "_attn/")
                crossAttn_f = crossAttn_dir + attn_dict_json[emo][model]["speechid"][show_txt] + ".npy"
                if model not in crossAttn_dict_for_vis.keys():
                    crossAttn_dict_for_vis[model] = {}
                if emo not in crossAttn_dict_for_vis.keys():
                        crossAttn_dict_for_vis[model][emo] = {}
                # attn, syn_durs, syn_phones, ref_durs, ref_phones
                crossAttn_dict_for_vis[model][emo] = (
                    np.load(crossAttn_f, allow_pickle=True)[show_t, show_b, batch_n, show_h, ...], # [time, block_n, batch, head, t_t, t_s]
                    attn_dict_json[emo][model]["k_dur"][show_txt],
                    attn_dict_json[emo][model]["phonemes"][show_txt],
                    attn_dict_json[emo][model]["q_dur"][show_txt],
                    attn_dict_json[emo][model]["phonemes"][show_txt].copy(),  ## Tempt
                )
    return crossAttn_dict_for_vis


def convert_attnEnh_json(attn_dict_json, attn_dir, show_t=0, show_b=0, show_h=0, show_txt=0):
    """
    For visualizing crossAttn
    - auxiliary line given mfa duration ?

    """
    batch_n = 0
    crossAttn_dict_for_vis = dict()  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}  <- choose t and h from attn_map_dict
    for model, psd in attn_dict_json.items():
        if model != "reference":
            crossAttn_dir = os.path.join(attn_dir, model + "_attn/")
            crossAttn_f = crossAttn_dir + attn_dict_json[model]["speechid"][show_txt] + ".npy"
            if model not in crossAttn_dict_for_vis.keys():
                crossAttn_dict_for_vis[model] = {}
            # attn, syn_durs, syn_phones, ref_durs, ref_phones

            crossAttn_dict_for_vis[model] = (
                np.load(crossAttn_f, allow_pickle=True)[show_t, show_b, batch_n, show_h, ...], # [time, block_n, batch, head, t_t, t_s]
                attn_dict_json[model]["k_dur"][show_txt],
                attn_dict_json[model]["phonemes"][show_txt],
                attn_dict_json[model]["q_dur"][show_txt],
                attn_dict_json[model]["phonemes"][show_txt].copy(),  ## Tempt
            )
    return crossAttn_dict_for_vis


def convert_attn_json_bk(prosody_dict_json, out_dir, show_t=0, show_b=0, show_h=0, show_txt=0):
    """
    For visualizing crossAttn
    - auxiliary line given mfa duration ?

    """
    batch_n = 0
    crossAttn_dict_for_vis = dict()  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}  <- choose t and h from attn_map_dict
    crossAttn_dict_for_vis_aux = dict()  # {"model1": {"emo1": ([pdurs_nums], [p_nums])}}}
    print("4. Vis attention map given attn file")

    for emo, m_psd in prosody_dict_json.items():
        for model, psd in m_psd.items():
            if model != "reference":
                crossAttn_dir = os.path.join(out_dir, model + "_attn/")
                crossAttn_f = crossAttn_dir + prosody_dict_json[emo][model]["speechid"][show_txt] + ".npy"
                if model not in crossAttn_dict_for_vis.keys():
                    crossAttn_dict_for_vis[model] = {}
                if emo not in crossAttn_dict_for_vis.keys():
                        crossAttn_dict_for_vis[model][emo] = {}

                # attn, syn_durs, syn_phones, ref_durs, ref_phones
                crossAttn_dict_for_vis[model][emo] = (
                    np.load(crossAttn_f, allow_pickle=True)[show_t, show_b, batch_n, show_h, ...], # [time, block_n, batch, head, t_t, t_s]
                    prosody_dict_json[emo][model]["duration"][show_txt],
                    prosody_dict_json[emo][model]["phonemes"][show_txt],
                    prosody_dict_json[emo]["reference"]["duration"][show_txt],
                    prosody_dict_json[emo]["reference"]["phonemes"][show_txt],
                )
    return crossAttn_dict_for_vis
