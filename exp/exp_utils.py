from pymcd.mcd import Calculate_MCD
import librosa
import soundfile as sf
# instance of MCD class three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
mcd_toolbox = Calculate_MCD(MCD_mode="MCD-DTW")
# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
#mcd_value = mcd_toolbox.calculate_mcd("001.wav", "002.wav")import librosa
from pymcd.mcd import Calculate_MCD
import torch
import tgt
import numpy as np
import os
from const_param import textgrid_dir
import sys, os, yaml, json
import numpy as np
from pathlib import Path
import torch
from const_param import emo_melstyleSpk_dict, config_dir, logs_dir, melstyle_dir, wav_dir, wav_dict, emo_num_dict, logs_dir_par, psd_quants_dir
import shutil
from exp_utils2 import parse_filelist, intersperse, get_emo_label
import pandas as pd
from utils import maximum_path, mask_from_lens, length_to_mask


##################### Operate file ##################

def convert_json_to_pd2(json_data, custom_order=None, need_multi_index=True, need_print_latex=True):
    """
    {"hyper":
        "model": [value1, value2]}
    """
    rows = []
    for hyper, model_values in json_data.items():
        for model, values in model_values.items():
            # pad or trim to two values if needed
            value1, value2 = (values + [None, None])[:2]
            rows.append({
                "model": model,
                "emotion": hyper,
                "pitch": value1,
                "energy": value2})
    # ---- Convert to DataFrame ----
    df = pd.DataFrame(rows)[["model", "emotion", "pitch", "energy"]]

    if custom_order is not None:
        df["model"] = pd.Categorical(df["model"], categories=custom_order, ordered=True)
        df = df.sort_values(["model", "emotion"]).reset_index(drop=True)


    # compute average
    avg_per_model = (
        df.groupby("model")[["pitch", "energy"]]
        .mean()
        .round(2)
    )

    # multi-index column
    if need_multi_index:
        df_wide = df.pivot(index='model', columns='emotion', values=['pitch', 'energy'])
        emotion_order = ['Angry', 'Neutral', 'Sad', 'Happy', 'Surprise']

        # reorder both levels
        df_wide = (
            df_wide
            .swaplevel(0, 1, axis=1)  # emotion → first level
            .reindex(columns=pd.MultiIndex.from_product(
                [emotion_order, ['pitch', 'energy']]
            ))  # enforce pitch first
        )
        """        
        df_wide = df_wide.reindex(columns=pd.MultiIndex.from_product(
            [['pitch', 'energy'], emotion_order]
        )).swaplevel(0, 1, axis=1)
        """
        df = df_wide.round(2)

    if need_print_latex:
        df_marked = df.copy()
        PRINT_BEST = True
        if PRINT_BEST:
            for col in df.columns:
                # get sorted unique values (descending = best first)
                sorted_vals = df[col].sort_values(ascending=True).unique()
                best = sorted_vals[0]
                second = sorted_vals[1] if len(sorted_vals) > 1 else None
                # apply formatting
                df_marked[col] = df[col].apply(
                    lambda x:
                    f"\\first{{{x}}}" if x == best else
                    (f"\\second{{{x}}}" if second is not None and x == second else f"{x}")
            )
        for idx, row in df_marked.iterrows():
            print(f"{idx} & " + " & ".join(row.astype(str)) + r" \\")
    return df, avg_per_model


def convert_json_to_pd(json_data):
    rows = []
    for model, hypers in json_data.items():
        for hyper, subhypers in hypers.items():
            for subhyper, values in subhypers.items():
                # pad or trim to two values if needed
                value1, value2 = (values + [None, None])[:2]
                rows.append({
                    "model": model,
                    "hyper": hyper,
                    "subhyper": subhyper,
                    "value1": value1,
                    "value2": value2})
    # ---- Convert to DataFrame ----
    df = pd.DataFrame(rows)
    return df


def combine_jsons(attn_model1_json, attn_model2_json, combined_model12_json):
    """
    combine two jsons
    Args:
        attn_model1_json (_type_): {"spk": {"emo": {"A/B": {"ids/qkdurs/qkphones":... }}}}
        attn_model2_json (_type_): _description_
        combined_model12_json (_type_): _description_
    """
    with open(attn_model1_json, 'r') as f:
        attn_model1 = json.load(f)
    with open(attn_model2_json, 'r') as f:
        attn_model2 = json.load(f)

    combined_dict = combine_two_jsons(attn_model1, attn_model2)

    with open(combined_model12_json, 'w') as f:
        json.dump(combined_dict, f, indent=4)

def combine_two_jsons(attn_model1, attn_model2):
    """
    attn_model1: {"spk": {"emo": {"A/B": {"ids/qkdurs/qkphones":... }}}}
    """
    combined_dict = attn_model1.copy()
    for spk, emo_model_dict in attn_model2.items():
        for emo, model_dict in emo_model_dict.items():
            for model, psd_dict in model_dict.items():
                if model not in combined_dict[spk][emo].keys():
                    combined_dict[spk][emo][model] = attn_model2[spk][emo][model]
    return combined_dict


def renew_dict(current_dict, old_dict):
    """ renew old dict with current dict (Only add model subdirectory of old dict when it is not in current dict ) 
    Args:
        current_dict (_type_): {"speaker": {"emo1": {"model1": {"speechid/qkdur/phone0": [[],[]]}}}}
        old_dict (_type_): {"speaker": {"emo1": {"model1": {"speechid/qkdur/phone0": [[],[]]}}}}
    Returns:
        _type_: _description_
    """
    new_dict = current_dict.copy()
    for spk, emo_model_dict in old_dict.items():    
        for emo, model_dict in emo_model_dict.items(): 
            for model, psd_dict in model_dict.items():
                if model not in new_dict[spk][emo].keys():
                    new_dict[spk][emo][model] = old_dict[spk][emo][model]
    return new_dict

def get_styles_wavs_from_dict(style_dict, emo_type="index", emo_num=5):
    styles = []
    for emo, mel_spk in style_dict.items():
        melstyle, spk = mel_spk
        emo_tensor = get_emo_label(emo, emo_num_dict, emo_type="index")

        spk_tensor = torch.LongTensor([spk]).cuda()
        melstyle_tensor = torch.from_numpy(np.load(
            melstyle_dir + "/" + melstyle)).cuda()
        melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)
        wav_n = os.path.join(wav_dir, melstyle.split(".")[0] + ".wav")
        styles.append((emo, spk, wav_n, emo_tensor, melstyle_tensor, spk_tensor))
    return styles

def get_synText_from_file(synText_f):
    """
    he was still in the forest!
    he was still in the forest!
    he was still in the forest!
    """
    with open(synText_f, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    return texts


def get_synStyle_from_file(synStyle_f,
                           split_char="|",
                           melstyle_type="codec",
                           wav_dir=wav_dir,
                           psd_quants_dir=psd_quants_dir,
                           melstyle_dir=melstyle_dir,
                           dataset_name="esd"
                           ):
    """
    get style related features from file with below format
    /home/rosen/data/ESD/0013/Angry/train/0013_000628.wav|I know you .
    psd_quants_dir: FACodec
    melstyle_dir:   wav2vec2
    """
    styles = []
    syn_styles = parse_filelist(synStyle_f, split_char=split_char)  # emotion changed

    if len(syn_styles[0]) == 5:
        for speechid, spk, phoneme, txt, emo in syn_styles:
            spk = int(spk[2:])
            wav_n = os.path.join(wav_dir, speechid + ".wav")
            emo_tensor = get_emo_label(emo, emo_num_dict, emo_type="index")
            spk_tensor = torch.LongTensor([spk]).cuda()
            if melstyle_type == "codec":
                melstyle_tensor = torch.from_numpy(np.load(psd_quants_dir + "/" + speechid + ".npy")).cuda()
            else:
                melstyle_tensor = torch.from_numpy(np.load(melstyle_dir + "/" + speechid + ".npy")).cuda()
                melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)
            styles.append((emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor))
    elif len(syn_styles[0]) == 6:
        for speechid, spk, phoneme, txt, syl_start, emo in syn_styles:
            spk = int(spk[2:])
            wav_n = os.path.join(wav_dir, speechid + ".wav")
            emo_tensor = get_emo_label(emo, emo_num_dict, emo_type="index")
            spk_tensor = torch.LongTensor([spk]).cuda()
            if melstyle_type == "codec":
                melstyle_tensor = torch.from_numpy(np.load(psd_quants_dir + "/" + speechid + ".npy")).cuda()
            else:
                melstyle_tensor = torch.from_numpy(np.load(melstyle_dir + "/" + speechid + ".npy")).cuda()
                melstyle_tensor = melstyle_tensor.unsqueeze(0).transpose(1, 2)
            syl_start = torch.tensor([int(i) for i in syl_start.split(",")], dtype=torch.long).unsqueeze(0)
            styles.append((emo, spk, wav_n, txt, emo_tensor, melstyle_tensor, spk_tensor, syl_start))
    elif len(syn_styles[0]) == 2:
        if dataset_name == "esd":
            for speech_path, txt in syn_styles:
                # search emotion index
                emo_elem_index = None
                emotions = ["Happy", "Angry", "Neutral", "Sad", "Surprise"]
                speech_path_elem = speech_path.split("/")
                for i, elem in enumerate(speech_path_elem):
                    if elem in emotions:
                        emo_elem_index = i
                spk_elem_index = emo_elem_index - 1
                spk = speech_path.split("/")[spk_elem_index]
                emo = speech_path.split("/")[emo_elem_index]
                wav_n = speech_path.split("/")[-1].split(".")[0]
                #psd = psd_code_path = os.path.join(psd_code_dir, f'{wav_n}.npy')
                styles.append((spk, emo, txt, speech_path))
        elif dataset_name == "libritts":
            for speech_path, txt in syn_styles:
                # search emotion index
                spk = speech_path.split("/")[6]
                emo = "Neutral"
                styles.append((spk, emo, txt, speech_path))
    return styles


def fine_adjust_configs(bone_n, bone_option, bone2configPath, config_dir):
    """
    fine-adjust model/train configs of bone_n by bone_option
    Returns:

    """
    log_dir_base = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/{}/"
    preprocess, model_config, train_config = bone2configPath[bone_n]
    preprocess_config = yaml.load(
        open(config_dir + "/" + preprocess, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        config_dir + "/" + model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        config_dir + "/" + train_config, "r"), Loader=yaml.FullLoader)
    if bone_n == "stditCross":
        if bone_option == "noguide" or bone_option == "base":
            model_config["stditCross"]["guide_loss"] = False
            model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = "frame"
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_base_codec")
        elif bone_option == "guideframe":
            model_config["stditCross"]["guide_loss"] = True
            model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = "frame"
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_guideLoss_codec")
        elif bone_option == "guidephone":
            model_config["stditCross"]["guide_loss"] = True
            model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = "phone"
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_guideLoss_codec_phoneRope")
        elif bone_option == "guideSyl":
            model_config["stditCross"]["guide_loss"] = True
            model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = "sel"
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_guideLoss_codec_sylRope")
        elif bone_option == "pguidephone":
            model_config["stditCross"]["guide_loss"] = True
            model_config["stditCross"]["decoder_config"]["stdit_config"]["phoneme_RoPE"] = "phone"
            train_config["path"]["log_dir"] = log_dir_base.format("stditCross_pguidephone_codec")
    return preprocess_config, model_config, train_config

def copy_ref_speech(src_wavs, src_txts, dst_wavs, dst_txts):
    for src_wav, src_txt, dst_wav, dst_txt in zip(src_wavs, src_txts, dst_wavs, dst_txts):
        shutil.copyfile(src_wav, dst_wav)
        with open(dst_txt, "w") as file1:
            # Writing data to a file
            file1.write(src_txt)

def copy_reference_speech(syn_styles, ref_dir):
    # copy reference
    ref_texts = []
    for i, ref_s in enumerate(syn_styles):
        spk, emo, ref_txt, speech_path = ref_s
        ref_texts.append(ref_txt) if ref_txt not in ref_texts else ref_texts
        r_id = ref_texts.index(ref_txt)
        r_wav_f = f'spk{spk}_{emo}_ref{r_id}.wav'
        shutil.copy(speech_path, os.path.join(ref_dir, r_wav_f))

def get_pitch_match_score(pitch1, pitch2):
    pitch_score = 0
    return pitch_score


def get_pitch_match_score(pitch1, pitch2):
    pitch_score = 0
    return pitch_score


def get_ref_pRange(p_start=5, p_end=8, ref_id="0019_000403"):
    """
    # S(5) T(6) IH1(7) L(8)
    get range of specific phoneme on frames by given phoneme index of reference speech id
    """
    # get the time/frame range of given ref text (Know from ref id)
    speaker = ref_id.split("_")[0]
    tg_path = os.path.join(
        textgrid_dir, speaker, "{}.TextGrid".format(ref_id)
    )
    textgrid = tgt.io.read_textgrid(tg_path)
    # duration = frame number
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones")
    )
    #pIndex = get_pIndex_from_wIndx(ref_wIndx) # ?? word ??
    ref_nRange = (sum(duration[:p_start]), sum(duration[:p_end+1]))
    #frame_n = sum(duration)
    # check if ref_nRange match the spefied frame (still) ??
    return ref_nRange


def get_tgt_pRange():
    pass

def get_pIndex_from_wIndx(ref_wIndx):
    pass


def get_alignment(tier):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    sampling_rate = 16000
    hop_length = 256

    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    return phones, durations, start_time, end_time

def compute_phoneme_mcd(wav1,
                        wav2,
                        p1_interv=(0, 1),
                        p2_interv=(0, 1),
                        ):
    # read wav1, wav2
    y1, sr1 = librosa.load(wav1, sr=None)
    y2, sr2 = librosa.load(wav2, sr=None)
    # split
    y1_split = y1[p1_interv[0] * sr1 : p1_interv[1]]
    y2_split = y2[p2_interv[0] * sr2 : p2_interv[1]]

    # write to file
    wav1_split = wav1.split(".")[0] + "_p{}_{}.wav".format(
        p1_interv[0], p1_interv[1])
    wav2_split = wav2.split(".")[0] + "_p{}_{}.wav".format(
        p2_interv[0], p2_interv[1])
    sf.write(wav1_split, y1_split, sr1)
    sf.write(wav2_split, y2_split, sr2)
    # mcd
    mcd_toolbox.calculate_mcd(wav1_split, wav2_split)

def convert_frameNum2second(frame_num, sr=16000, chunk_len=8000):
    return frame_num * chunk_len / sr


############ vis related ############
def convert_xydur_xybox(x_dur, y_dur):
    if len(x_dur)!=len(y_dur):
        IOError("xdur and ydur should be same length: {}, {}".format(len(x_dur), len(y_dur)))
    xywh_list = []
    for i, (xd, yd) in enumerate(zip(x_dur, y_dur)):
        if i < len(x_dur)-1:
            x, y = int(xd), int(yd)
            w, h = int(x_dur[i+1] - xd), int(y_dur[i+1] - yd)
            xywh_list.append((x, y, w, h))
    return xywh_list

def clean_phone(phone, phone_durs, internum=11):
    """
    remove "" and internum, append their dur to latter phoneme
    Args:
        phones: ["", "AH", "", 11, "", "B", "", "AH0", "", "IH, "", 11, "", "AH", "", "G", ""]
        phone_durs: [2, 3, 5, 6, ...]
    Returns:
        syls: ["AH", "B", ... , "G"]
        syl_durs  [5, 11, ..., ...]
    """
    assert len(phone) % 2 == 1  # phone must be odd number
    assert len(phone) == len(phone_durs)
    unintersperse_phone = [str(phone[pos]) if phone[pos] != internum else "" for pos in range(1, len(phone), 2)]
    unintersperse_phone_durs = [phone_durs[pos] + phone_durs[pos-1] for pos in range(1, len(phone_durs), 2)]
    unintersperse_phone_durs[-1] += phone_durs[-1]  # add last phone dur
    return unintersperse_phone, unintersperse_phone_durs

def cut_pad_a2b_len_left(a, b_length):
    if a.size > b_length:  # cut from start
        cut_start = a.size - b_length
        a = a[cut_start:]
    elif a.size < b_length:  # pad from start
        a = pad_a2b_left(a, b_length)
    else:
        pass
    return a

def pad_a2b_left(a, b_length):
    """pad a to length same as b"""
    a_new = np.zeros([b_length], device=a.device)
    pad_end = b_length - a.size
    a_new[:pad_end] = a[:pad_end]
    a_new[pad_end:] = a
    return a_new


def save_attn_dict(attn_dict, nested_keys, values):
    spk, emo, model_n = nested_keys
    speechid, syn_phonemes, ref_phonemes, q_dur, k_dur = values
    spk = "spk" + str(spk)
    if spk not in attn_dict.keys():
        attn_dict[spk] = dict()
    if emo not in attn_dict[spk].keys():
        attn_dict[spk][emo] = dict()
    if model_n not in attn_dict[spk][emo].keys():
        attn_dict[spk][emo][model_n] = {"speechid": [], "syn_phonemes": [], "ref_phonemes": [], "q_dur": [], "k_dur": []}
    attn_dict[spk][emo][model_n]["speechid"].append(speechid)
    attn_dict[spk][emo][model_n]["syn_phonemes"].append(syn_phonemes)
    attn_dict[spk][emo][model_n]["ref_phonemes"].append(ref_phonemes)
    attn_dict[spk][emo][model_n]["q_dur"].append(q_dur)
    attn_dict[spk][emo][model_n]["k_dur"].append(k_dur)
    return attn_dict


def save_psdcond(psdcond_dict, nested_keys, values):
    spk, emo, model_n = nested_keys
    speechid, pitch_cond, energy_cond = values
    spk = "spk" + str(spk)
    if spk not in psdcond_dict.keys():
        psdcond_dict[spk] = dict()
    if emo not in psdcond_dict[spk].keys():
        psdcond_dict[spk][emo] = dict()
    if model_n not in psdcond_dict[spk][emo].keys():
        psdcond_dict[spk][emo][model_n] = {"speechid": [], "pitch_cond": [], "energy_cond": []}
    psdcond_dict[spk][emo][model_n]["speechid"].append(speechid)
    psdcond_dict[spk][emo][model_n]["pitch_cond"].append(pitch_cond)
    psdcond_dict[spk][emo][model_n]["energy_cond"].append(energy_cond)
    return psdcond_dict

def extract_k_dur(ref_mel, ref_phone_tokens, second_model, device="cuda"):
    # calcuate dur of referece speech
    ref_mel_len = torch.tensor([ref_mel.size(-1)]).to(device)
    ref_mel_mask = length_to_mask(ref_mel_len // 2).to(device)
    _, _, s2s_attn = second_model.text_aligner(ref_mel, ref_mel_mask, ref_phone_tokens)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)
    phoneme_len = torch.tensor([ref_mel.size(-1)]).to(device)
    mask_ST = mask_from_lens(s2s_attn, phoneme_len, ref_mel_len)
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    k_dur = s2s_attn_mono.squeeze().sum(-1)
    return k_dur


if __name__ == '__main__':
    speech_dir = ("/home/rosen/Project/Speech-Backbones/GradTTS/logs/interpEmoTTS_frame2binAttn_noJoint/"
                  "chpt400_time50_spk19_emoAngry")
    non_p2p = speech_dir + "/no_p2p_sample_v1_0.wav"
    p2p = speech_dir + "/p2p_sample_v1_0.wav"

    origin_dir = "/home/rosen/Project/FastSpeech2/ESD/16k_wav"
    origin = origin_dir + "/0019_000403.wav"

    mcd_origin_nonP2P = compute_phoneme_mcd(
         non_p2p,
        origin,
        p1_interv=(0, 1),
        p2_interv=(0, 1)
    )

    mcd_origin_p2p = compute_phoneme_mcd(
        p2p,
        origin,
        p1_interv=(0, 1),
        p2_interv=(9, 1)
    )
    print("mcd betwenn origin and nonp2p is: {}".format(mcd_origin_nonP2P))
    print("mcd betwenn origin and p2p is: {}".format(mcd_origin_p2p))


