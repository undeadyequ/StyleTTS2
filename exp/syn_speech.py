import sys, os, yaml, json
from itertools import accumulate

import torch
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
import shutil

from api_cfm import StableTTSAPI, text2interphone, text2interphonetensor
from api_ddpm_self import StableTTSAPI as StableTTSAPISELF
import torchaudio
from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2
from text import cleaned_text_to_sequence
from datas.dataset import intersperse

## CONSTANTS
g2p_mapping = {
    'chinese': chinese_to_cnm3,
    'japanese': japanese_to_ipa2,
    'english': english_to_ipa2,
}
lang = "english"
g2p = g2p_mapping.get(lang)


def syn_speech_from_model_batch(model1_n_tts_voc, styles, synTexts, style_syntex_name, out_dir, inference_config=None, save_attn_json_file=False, save_attn=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name, model_config, chk_pt, mel_config, voc_config, vocoder_model_path = model1_n_tts_voc
    model = StableTTSAPI(model_name, chk_pt, vocoder_model_path, model_config, mel_config, voc_config, 'vocos').to(device)
    sr = mel_config.sample_rate

    out_speech_dir = os.path.join(out_dir, model_name, style_syntex_name)
    out_attn_dir = os.path.join(out_dir, model_name, style_syntex_name + "_attn")
    ref_speech_dir = os.path.join(out_dir, "reference", style_syntex_name)
    attn_json_path = os.path.join(out_dir, "attn_{}_{}.json".format(model_name, style_syntex_name))  # only needed in random synthesis

    if not os.path.isdir(out_speech_dir):
        Path(out_speech_dir).mkdir(exist_ok=True, parents=True)
    if not os.path.isdir(ref_speech_dir):
        Path(ref_speech_dir).mkdir(exist_ok=True, parents=True)
    if not os.path.isdir(out_attn_dir):
        Path(out_attn_dir).mkdir(exist_ok=True, parents=True)

    ref_texts = []
    attn_dict = {}  # for record qk_dur and phoneme for crossAttn visualization
    for j, style in enumerate(styles):
        spk, emo, ref_txt, speech_path = style
        ref_texts.append(ref_txt) if ref_txt not in ref_texts else ref_texts
        r_id = ref_texts.index(ref_txt)
        speech_paths = [speech_path] * len(synTexts)  # repeat style for each text
        print("synthesize {} speech by {} given style {}".format(len(synTexts), model_name, speech_path))
        audio_outputs, _, crossAttn, q_durs, k_durs = model.inference_batch(synTexts, speech_paths, **inference_config, ref_txt=ref_txt)
        # save wav, attn, txt

        # copy reference
        ref_speech_id = f'spk{spk}_{emo}_ref{r_id}'
        src_ref_txt_f = f'{ref_speech_dir}/{ref_speech_id}.lab'
        with open(src_ref_txt_f, "w") as file1:
            file1.write(ref_txt)
        dst_wav = f'{ref_speech_dir}/{ref_speech_id}.wav'
        shutil.copyfile(speech_path, dst_wav)

        for k, (text, audio_output) in enumerate(zip(synTexts, audio_outputs)):
            speech_id = f'spk{spk}_{emo}_ref{r_id}_syn{k}'
            wav_n = f'{out_speech_dir}/{speech_id}.wav'
            txt_f = f'{out_speech_dir}/{speech_id}.lab'
            attn_f = f'{out_attn_dir}/{speech_id}.npy'
            torchaudio.save(wav_n, audio_output, sr)
            with open(txt_f, "w") as file1:
                file1.write(text)

            # save attn
            if save_attn and crossAttn is not None:
                np.save(attn_f, crossAttn[:, k, ...].cpu().numpy())  ## [(time), block_n, batch, head, t_t, t_s]

            if save_attn_json_file:
                q_dur_zero_indexes = (q_durs[k] == 0).nonzero(as_tuple=True)[0]
                if len(q_dur_zero_indexes) != 0:
                    q_dur_len = q_dur_zero_indexes[0]
                else:
                    q_dur_len = len(q_durs[k])
                q_dur = q_durs[k][:q_dur_len]  # q_dur exclude padding
                k_dur = k_durs[k]              # k_dur is duplicated, therefore no padding
                syn_phonemes = text2interphone(g2p, text)
                ref_phonemes = text2interphone(g2p, ref_txt)
                attn_dict = save_attn_dict(
                    attn_dict,
                    (spk, emo, model_name),
                    (speech_id, syn_phonemes, ref_phonemes, q_dur.squeeze(0).cpu().numpy().tolist(), k_dur.squeeze(0).cpu().numpy().tolist())
                )
        if save_attn_json_file:
            with open(attn_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(attn_dict, sort_keys=True, indent=4))



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