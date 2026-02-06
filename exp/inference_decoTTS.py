import os.path
import shutil
import time
from itertools import accumulate

import torch
import sys

from soxr import resample

sys.path.append("../")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch, maximum_path, mask_from_lens, log_norm, append_sentence_to_file, length_to_mask
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import phonemizer
from Utils.PLBERT.util import load_plbert
from exp_utils_bk import get_synText_from_file, get_synStyle_from_file
from exp_utils import save_attn_dict, extract_k_dur, save_psdcond
from pathlib import Path
from load_vocoder import get_vocoder
import os
import nltk
from ref_aware_pe2 import build_voiced_mask, fill_unvoiced_with_interp, fuse_prosody_final
from ref_aware_pe3 import fuse_prosody_smooth_additive
from vis2 import plot_f0_comparison
from exec_utmosv2 import run_utmos
from inference_batch import inference_batch
from utils import frame_to_phoneme_avg_and_back_binary, align_dur2

nltk.download('punkt_tab')
os.chdir('/home/rosen/Project/StyleTTS2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = "cpu"

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

generator = get_vocoder(ckpt_dir="/home/rosen/ckpt/styletts/Vocoder/LibriTTS/", device=device)


# GET model
def get_pretrained_modules():
    textclenaer = TextCleaner()
    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
    return global_phonemizer, textclenaer

def get_second_model(ckpt="/home/rosen/ckpt/styletts2_libriTTS/epochs_2nd_00020.pth",
                     config_f="/home/rosen/ckpt/styletts2_libriTTS/config.yml",
                     model_name="styletts2_txt2mel"):
    # load phonemizer
    config = yaml.safe_load(open(config_f))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # build model
    model_params = recursive_munch(config['model_params'])

    if "styletts2_txt2mel" in model_name:
        from models_txt2mel import build_model
        model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    elif "mdit_cfm" in model_name:
        from models_txt2mel_cfm import build_model
        model = build_model(model_params, config['cfm_config'], text_aligner, pitch_extractor, plbert)
    else:
        print(f"no such model: {model_name}")

    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # load ckpt
    params_whole = torch.load(ckpt, map_location='cpu')
    params = params_whole['net']
    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])

    _ = [model[key].eval() for key in model]

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
        clamp=False)

    return model, sampler, model_params

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, model):
    mel_tensor = extract_mel_from_wav(path)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)

def extract_mel_from_wav(wav, sr=24000):
    """Must convert to 24000"""
    #waveform, sr = torchaudio.load(wav)
    wave, sr = librosa.load(wav, sr=sr)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    # cut to 1/2
    mel_len = mel_tensor.size(-1)
    acoustic_feature = mel_tensor[:, :, :(mel_len - mel_len % 2)]  # only even
    return acoustic_feature

def inference_first(text, ref_wav, model, out_wav_f):
    # process text
    global_phonemizer, textclenaer = get_pretrained_modules()
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        # text and mel
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        mels = extract_mel_from_wav(ref_wav)
        mel_input_length = mels.size(-1)  # CHECK
        mel_input_length = torch.Tensor([mel_input_length + 1, ]).to("cuda")

        # dur
        n_down = model.text_aligner.n_down
        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
        _, _, s2s_attn = model.text_aligner(mels, mask, tokens)
        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)
        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # asr = text * dur
        asr = (t_en @ s2s_attn_mono)

        F0_real, _, _ = model.pitch_extractor(mels.unsqueeze(1))
        real_norm = log_norm(mels.unsqueeze(1)).squeeze(1)
        s = model.style_encoder(mels.unsqueeze(1))

        end = int(mel_input_length[0]//2)
        mel_rec = model.decoder(asr[:, :end], F0_real, real_norm, s)

        # add vocoder
        c_pred = mel_rec.squeeze()
        y_pred = generator(c_pred.unsqueeze(0))

        torchaudio.save(out_wav_f, y_pred.squeeze(0).cpu()[..., :-50], sample_rate=24000)
    return mel_rec.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later


@torch.no_grad()
def inference_second_hierstyle(text, ref_wav, model, sampler,
                     alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1, cfg_strength=None):
    """
    args:
        alpha=0.3: weight of acoustic style using styleDiff against styleEnc.
        beta=0.7: weight of prosodic style using styleDiff against styleEnc.
        pe_type:  "ref_pe" or "pred_pe"
    """
    style_dim = 256
    #start_time = time.time()
    # process text
    global_phonemizer, textclenaer = get_pretrained_modules()
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    # process wav
    ref_mel = extract_mel_from_wav(ref_wav)
    ref_s = compute_style(ref_wav, model)

    with (torch.no_grad()):
        # mask
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        output_lengths = torch.LongTensor([ref_mel.shape[-1]]).to(device)
        #mel_mask = length_to_mask(output_lengths).to(device)

        # text encoder
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        # style encoding
        s_pred = sampler(noise=torch.randn((1, style_dim)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=embedding_scale,
                         features=ref_s,  # reference from the same speaker as the embedding
                         num_steps=diffusion_steps).squeeze(1)
        if torch.isnan(s_pred[0, 0]):
            s = ref_s[:, int(style_dim/2):]
            ref = ref_s[:, :int(style_dim/2)]
        else:
            s = beta * s_pred[:, int(style_dim/2):] + (1 - beta) * ref_s[:, int(style_dim/2):]     # alpha/beta == 1: Use diffusion. (==0, use style encoding)
            #s_prosody = ref_s[:, int(style_dim/2):]
            ref = alpha * s_pred[:, :int(style_dim/2)] + (1 - alpha) * ref_s[:, :int(style_dim/2)]

        # dur prediction by psd style encoding
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))

        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # semantic embedding
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        #if model_params.decoder.type == "hifigan":   # why no need first one?
        if True:
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        # pred and preprocessed gd prosody.
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        F0_ref, _, F0 = model.pitch_extractor(ref_mel.unsqueeze(1))
        N_ref = log_norm(ref_mel.unsqueeze(1)).squeeze(1)

        # preprocess gd prosody (get s2s, phoneme-average, u/v masking)
        s2s = get_s2s(tokens, input_lengths, ref_mel, output_lengths, model.text_aligner)
        pe = torch.cat([N_pred.unsqueeze(1), F0_pred.unsqueeze(1)], dim=1)
        s2s = F.interpolate(s2s, size=F0_ref.size(-1), mode="nearest")
        _, _, N_real_phone, F0_real_phone = frame_to_phoneme_avg_and_back_binary(N_ref, F0_ref, s2s)
        pe_real = torch.cat([N_real_phone[None, ...], F0_real_phone[None, ...]], dim=1)
        pe_voice_mask = (pe[:, 1, :] >= 50.0) & (pe[:, 1, :] <= 600.0) & torch.isfinite(pe[:, 1, :])
        pe_real = align_dur2(pe_real, pe_voice_mask.unsqueeze(1), pitch_idx=1)

        # txt embedding
        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))

        # syn speech
        asr_new = torch.zeros_like(asr)  # Why this?
        asr_new[:, :, 0] = asr[:, :, 0]
        asr_new[:, :, 1:] = asr[:, :, 0:-1]
        asr = asr_new

        mel_rec, attn_maps = model.decoder(mu=asr, mask=None, n_timesteps=200, temperature=1.0, c=ref, seq_style=pe, p_mask=None,
                                           seq_style_gt=pe_real, cfg_strength=cfg_strength)

        # save mel
        c = mel_rec.squeeze()
        out = generator(c.unsqueeze(0))
        out = out.squeeze().cpu().numpy()[..., :-50]  # last 50 sounds not good

        return out, (F0_ref.squeeze(), F0_pred.squeeze(), pe_real[0, 1, ...].squeeze()), (N_ref.squeeze(), N_pred.squeeze(), pe_real[0, 0, ...].squeeze()), attn_maps, pred_dur  # weird pulse at the end of the model, need to be fixed later

def syn_speech_by_second_model(synTexts, syn_styles, out_dir, second_model, sampler,
                               alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1,
                               Vis_F0=False, reference_dir="", cfg_strength=None,
                               save_attn=False, save_cond=False, model_name="monoDiT", attn_filter=("0019", "Surprise")):
    """
    synthesize speech by synTexts and syn_styles
    """
    # ref_dir, out_dir, *_vis, *_attn
    if len(reference_dir) > 0:
        ref_dir = os.path.join(os.path.dirname(out_dir), reference_dir)
        if not os.path.isdir(ref_dir):
            Path(ref_dir).mkdir(exist_ok=True, parents=True)
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir(exist_ok=True, parents=True)
    if Vis_F0:
        vis_dir = out_dir + "_vis"
        if not os.path.isdir(vis_dir):
            Path(vis_dir).mkdir(exist_ok=True, parents=True)
    if not os.path.isdir(f'{out_dir}_attn') and save_attn:
        Path(f'{out_dir}_attn').mkdir(exist_ok=True, parents=True)

    ref_texts = []
    attn_json = {}
    psdcond_json = {}
    for i, ref_s in enumerate(syn_styles):
        spk, emo, ref_txt, speech_path = ref_s
        ref_texts.append(ref_txt) if ref_txt not in ref_texts else ref_texts
        r_id = ref_texts.index(ref_txt)
        #ref_s = compute_style(speech_path, second_model)

        # copy reference
        if len(reference_dir) > 0:
            r_wav_f = f'spk{spk}_{emo}_ref{r_id}.wav'
            shutil.copy(speech_path, os.path.join(ref_dir, r_wav_f))

        for k, text in enumerate(synTexts):
            #torch.manual_seed(0)   # IMPORTANT: global manual seed can not make initiate noises same to different models
            audio_output, (F0_ref, F0_pred, F0_align), (N_ref, N_pred, N_align), attn_maps, pred_dur = inference_second_hierstyle(
                                            text, speech_path, second_model, sampler,
                                            alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale,
                                            cfg_strength=cfg_strength)  # add model

            if isinstance(audio_output, tuple):
                audio_output = audio_output[0]
            speech_id = f'spk{spk}_{emo}_ref{r_id}_syn{k}'
            wav_n = f'{out_dir}/{speech_id}.wav'
            txt_f = f'{out_dir}/{speech_id}.lab'
            audio_output = torch.from_numpy(audio_output).unsqueeze(0)
            torchaudio.save(wav_n, audio_output, 24000)
            with open(txt_f, "w") as file1:
                file1.write(text)

            if Vis_F0:
                png_id = speech_id.split(".")[0]
                plot_f0_comparison(F0_ref, F0_pred, F0_align, out_path=f"{vis_dir}/{png_id}_f0.png")
                plot_f0_comparison(N_ref, N_pred, N_align, out_path=f"{vis_dir}/{png_id}_eng.png")

            if save_cond:
                psdcond_json = save_psdcond(psdcond_json,(spk, emo, model_name),
                                            (speech_id, F0_pred.cpu().squeeze(0).numpy().tolist(), N_pred.squeeze(0).cpu().numpy().tolist()))

            if save_attn:
                from utilities.vis import save_plot
                # Save attn_map
                if attn_filter is not None:
                    if attn_filter[0] in speech_id and attn_filter[1] in speech_id:
                        attn_path = f'{out_dir}_attn/{speech_id}.npy'
                        np.save(attn_path, attn_maps.cpu().numpy())  ## [time, block_n, batch, head, t_t, t_s]
                        #save_plot(attn_maps[0, 0, 0].detach().cpu(), "attn.png")
                else:
                    attn_path = f'{out_dir}_attn/{speech_id}.npy'
                    np.save(attn_path, attn_maps.cpu().numpy())  ## [time, block_n, batch, head, t_t, t_s]

                # Get syn_phones, ref_phonemes, q_dur, k_dur
                syn_phonemes, _ = get_phn(text)
                ref_phonemes, ref_phone_tokens = get_phn(ref_txt)
                q_dur = pred_dur
                ref_mel = extract_mel_from_wav(speech_path)
                k_dur = extract_k_dur(ref_mel, ref_phone_tokens, second_model, device)

                attn_json = save_attn_dict(
                    attn_json,
                    (spk, emo, model_name),
                    (speech_id, syn_phonemes, ref_phonemes, q_dur.squeeze(0).cpu().numpy().tolist(),
                     k_dur.squeeze(0).cpu().numpy().tolist()))
    return attn_json, psdcond_json


def get_phn(text):
    # phonemize
    global_phonemizer, textclenaer = get_pretrained_modules()
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])

    # tokenize
    ps_str = ' '.join(ps)
    tokens = textclenaer(ps_str)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    return ps, tokens

@torch.no_grad()
def get_s2s(texts, input_lengths, mels, mel_input_length, text_aligner):
    """
    :param
    texts:  (B, T, D)
    mels:  (B, T, D) (T, D)
    text_aligner:
    """
    mask = length_to_mask(mel_input_length // 2).to('cuda')
    text_mask = length_to_mask(input_lengths).to(texts.device)
    _, _, s2s_attn = text_aligner(mels, mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)
    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // 2)
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    return s2s_attn_mono


if __name__ == '__main__':
    model_root_dir = "/home/rosen/ckpt/styletts2_libriTTS/"
    model_config = {
        "styletts2_txt2mel": ["first_txt2mel/epoch_1st_00048.pth", "first_txt2mel/config_libritts_txt2mel.yml",
                              "first_txt2mel/epoch_2nd_00028.pth", "first_txt2mel/config_libritts_txt2mel.yml"],
        "mdit_cfm_v10": ["", "",
                        "first_txt2mel_cfm_v10/epoch_2nd_00048.pth",
                        "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "mdit_cfm_v10_epoch18": ["", "",
                         "first_txt2mel_cfm_v10/epoch_2nd_00018.pth",
                         "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "mdit_cfm_v16": ["", "",
                         "first_txt2mel_cfm_v16/epoch_2nd_00020.pth",
                         "first_txt2mel_cfm_v16/config_libritts_txt2mel_cfm_v16.yml"],
        "mdit_cfm_v17": ["", "",
                        "first_txt2mel_cfm_v17/epoch_2nd_00018.pth",
                        "first_txt2mel_cfm_v17/config_libritts_txt2mel_cfm_v17.yml"],
    }
    torch.manual_seed(0)

    # inference config
    dataset = "esd"  # libritts esd
    fuse_beta = 0.2
    cfg_strength = 3
    alpha = 0.3  # acoustic style. 1. use diffusion, 0: use encoding
    beta = 0.7  # prosodic style. 1. use diffusion, 0: use encoding

    ### IN
    model_name = "mdit_cfm_v10"  # "styletts2_txt2mel"  "mdit_cfm"  "mdit_cfm_v10"
    style = "exp/data/r1_test.txt"  # r1_50 r1_target libri_r1, r1_test, r1_sharp_last
    txt = "exp/data/s1_test.txt"  # s1_5 s1_target  libri_s1, s1_test, s1_10

    syn_styles = get_synStyle_from_file(style, split_char='|', dataset_name=dataset)  # emotion changed
    synTexts = get_synText_from_file(txt)

    # Get model
    second_model_path, second_config = model_root_dir + model_config[model_name][2], model_root_dir + model_config[model_name][3]
    second_model, sampler, model_params = get_second_model(ckpt=second_model_path, config_f=second_config, model_name=model_name)

    # OUT
    test_type = "hierstyle_test"  # monoStyle_compare  fuse_cond_test
    epoch_n = second_model_path.split(".")[0][-2:]
    alpha_str = str(alpha).replace(".", "")
    beta_str = str(beta).replace(".", "")
    out_dir = f"res/{test_type}/{model_name}_epoch{epoch_n}_{dataset}_{alpha_str}_{beta_str}"

    # run model
    syn_speech_by_second_model(synTexts, syn_styles, out_dir, second_model=second_model, sampler=sampler, reference_dir=f"reference_{dataset}", # in/out/model
                               alpha=alpha, beta=beta,                           # infer_config_fix (alpha, beta)
                               cfg_strength=cfg_strength,                        # infer_config_control (fuse_mono_cfg)
                               Vis_F0=False, save_attn=True, attn_filter=None)  # output_pattern
    # Save utmos_v2 score
    mean_mos, std_mos, mos_list = run_utmos(out_dir, 1)
    print(f"\nAverage UTMOS-v2: {mean_mos:.4f} ± {std_mos:.4f}")
    append_sentence_to_file("exp/utmosv2_log.txt", f"Average UTMOS-v2 of {out_dir}: {mean_mos:.4f} ± {std_mos:.4f}")
