import os.path
import shutil

import torch
import sys

from soxr import resample

sys.path.append("../")

torch.manual_seed(0)
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
from utils import recursive_munch, maximum_path, mask_from_lens, log_norm
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import phonemizer
from Utils.PLBERT.util import load_plbert
from exp_utils_bk import get_synText_from_file, get_synStyle_from_file
from pathlib import Path
from load_vocoder import get_vocoder
import os
import nltk
from ref_aware_pe2 import build_voiced_mask, fill_unvoiced_with_interp, fuse_prosody_final
from vis2 import plot_f0_comparison

nltk.download('punkt_tab')

os.chdir("..")
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

def get_first_model(ckpt="/home/rosen/ckpt/styletts2_libriTTS/first_txt2mel/epoch_1st_00048.pth",
                    config_f="/home/rosen/ckpt/styletts2_libriTTS/first_txt2mel/config_libritts_txt2mel.yml"):
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
    return model

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

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

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
    return mel_tensor

def inference_first(text, ref_wav, model, out_wav_f):
    # process text
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

def inference_second(text, ref_wav, model, sampler, model_params,
                     alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1,
                     wav_n=None, save_sr=24000, pe_type="", style_dim=256):
    """
    args:

        alpha=0.3: weight of acoustic style using styleDiff against styleEnc.
        beta=0.7: weight of prosodic style using styleDiff against styleEnc.
        pe_type:  "ref_pe" or "pred_pe"
    """
    # process text
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
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        if alpha == 0 and beta == 0:
            s = ref_s[:, int(style_dim/2):]
            ref = ref_s[:, :int(style_dim/2)]
        else:
            s_pred = sampler(noise=torch.randn((1, style_dim)).unsqueeze(1).to(device),
                             embedding=bert_dur,
                             embedding_scale=embedding_scale,
                             features=ref_s,  # reference from the same speaker as the embedding
                             num_steps=diffusion_steps).squeeze(1)
            s = s_pred[:, int(style_dim/2):]
            ref = s_pred[:, :int(style_dim/2)]

            s = beta * s + (1 - beta) * ref_s[:, int(style_dim/2):]
            ref = alpha * ref + (1 - alpha) * ref_s[:, :int(style_dim/2)]

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

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        # Choose PE
        if pe_type == "ref_pe":
            F0_cond, _, _ = model.pitch_extractor(ref_mel.unsqueeze(1))
            N_cond = log_norm(ref_mel.unsqueeze(1)).squeeze(1).detach()
        elif pe_type == "pred_pe":
            F0_cond, N_cond = model.predictor.F0Ntrain(en, s)
        elif pe_type == "ref_aware_pred_pe":
            F0_ref, _, _ = model.pitch_extractor(ref_mel.unsqueeze(1))
            N_ref = log_norm(ref_mel.unsqueeze(1)).squeeze(1).detach()
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            # get fused input
            ps_list = [p for p in ps]
            ps_list.insert(0, " ")
            voiced_mask = build_voiced_mask(ps_list, pred_aln_trg)
            voiced_mask_upsampled = F.interpolate(voiced_mask.unsqueeze(0).unsqueeze(0), scale_factor=2, mode='nearest').squeeze()
            F0_cond, N_cond, alpha = fuse_prosody_final(F0_pred.squeeze(), N_pred.squeeze(), F0_ref.squeeze(), N_ref.squeeze(), voiced_mask_upsampled, tau=0.15)

            # vis F0_ref, F0_pred, F0_fused
            png_id = text[:4] + "" + ref_wav[-8:-4]
            plot_f0_comparison(F0_ref.squeeze(), F0_pred.squeeze(), F0_cond, out_path=f"vis_f0_{png_id}.png")
        else:
            raise IOError("wrong pe_type")


        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))

        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        if model_params.decoder.type == "hifigan":
            if asr.size(-1) * 2 != F0_cond.size(-1): # need interpolate
                F0_cond = F.interpolate(F0_cond.unsqueeze(0), size=asr.size(-1) * 2, mode="linear", align_corners=True).squeeze(0)
                N_cond = F.interpolate(N_cond.unsqueeze(0), size=asr.size(-1) * 2, mode="linear", align_corners=True).squeeze(0)
            mel_rec = model.decoder(asr, F0_cond.unsqueeze(0), N_cond.unsqueeze(0), ref.squeeze().unsqueeze(0))
        elif model_params.decoder.type == "mdit_cfm":
            pe = torch.cat([N_cond.unsqueeze(1), F0_cond.unsqueeze(1)], dim=1)
            mel_rec, _ = model.decoder(mu=asr, mask=None, n_timesteps=200, temperature=1.0, c=ref, seq_style=pe,
                                       p_mask=None)
        else:
            print(f"{model_params.decoder.type} is not wrong")

        c = mel_rec.squeeze()
        out = generator(c.unsqueeze(0))

        #audio_output = torch.from_numpy(out).unsqueeze(0)
        # --- resample from 24kHz → 16kHz ---
        #resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=sr)
        #audio_output = resampler(audio_output)

        if wav_n is not None:
            torchaudio.save(wav_n, out.squeeze(0).cpu()[..., :-50], sample_rate=save_sr)
        return out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later
    #return out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later


def syn_speech_by_second_model(synTexts, syn_styles, out_dir, second_model, sampler, model_params, pe_type="pred_pe", alpha=0.3, beta=0.7, style_dim=256):
    # copy ref_dir
    ref_dir = os.path.join(os.path.dirname(out_dir), "reference")
    if not os.path.isdir(ref_dir):
        Path(ref_dir).mkdir(exist_ok=True, parents=True)

    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir(exist_ok=True, parents=True)


    ref_texts = []
    for i, ref_s in enumerate(syn_styles):
        spk, emo, ref_txt, speech_path = ref_s
        ref_texts.append(ref_txt) if ref_txt not in ref_texts else ref_texts
        r_id = ref_texts.index(ref_txt)
        #ref_s = compute_style(speech_path, second_model)

        # copy reference
        r_wav_f = f'spk{spk}_{emo}_ref{r_id}.wav'
        shutil.copy(speech_path, os.path.join(ref_dir, r_wav_f))

        for k, text in enumerate(synTexts):
            audio_output = inference_second(text, speech_path, second_model, sampler, model_params,
                                            alpha=alpha, beta=beta, diffusion_steps=10, embedding_scale=1, pe_type=pe_type, style_dim=style_dim)  # add model
            if isinstance(audio_output, tuple):
                audio_output = audio_output[0]
            speech_id = f'spk{spk}_{emo}_ref{r_id}_syn{k}'
            wav_n = f'{out_dir}/{speech_id}.wav'
            txt_f = f'{out_dir}/{speech_id}.lab'
            audio_output = torch.from_numpy(audio_output).unsqueeze(0)
            ## --- resample from 24kHz → 16kHz ---
            #resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=sr)
            #audio_output = resampler(audio_output)
            torchaudio.save(wav_n, audio_output, 24000)
            with open(txt_f, "w") as file1:
                file1.write(text)

if __name__ == '__main__':
    # Get model
    global_phonemizer, textclenaer = get_pretrained_modules()

    model_root_dir = "/home/rosen/ckpt/styletts2_libriTTS/"
    model_config = {
        "styletts2_txt2mel": ["first_txt2mel/epoch_1st_00048.pth", "first_txt2mel/config_libritts_txt2mel.yml",
                              "first_txt2mel/epoch_2nd_00028.pth", "first_txt2mel/config_libritts_txt2mel.yml"],
        "mdit_cfm": ["first_txt2mel_cfm/epoch_1st_00048.pth", "first_txt2mel_cfm/config_libritts_txt2mel_cfm.yml",
                     "first_txt2mel_cfm/epoch_2nd_00028.pth", "first_txt2mel_cfm/config_libritts_txt2mel_cfm.yml"],
        "mdit_cfm_v2": ["first_txt2mel_cfm_v2/epoch_1st_00048.pth", "first_txt2mel_cfm_v2/config_libritts_txt2mel_cfm.yml",
                     "first_txt2mel_cfm_v2/epoch_2nd_00028.pth", "first_txt2mel_cfm_v2/config_libritts_txt2mel_cfm.yml"],
    }
    #first_model_path, first_config = "/home/rosen/ckpt/styletts2_libriTTS/first_txt2mel/epoch_1st_00048.pth", "/home/rosen/ckpt/styletts2_libriTTS/first_txt2mel/config_libritts_txt2mel_first.yml"
    second_model_path, second_config = "/home/rosen/ckpt/styletts2_libriTTS/first_txt2mel/epoch_2nd_00028.pth", "/home/rosen/ckpt/styletts2_libriTTS/first_txt2mel/config_libritts_txt2mel.yml"

    TEST_FIRST_MODEL = False ## NO duration prediction
    TEST_SECOND_MODEL = False
    TEST_TEXT = True
    # test single speech
    text1 = "Please not go that path."
    ref_wav1 = "/home/rosen/Project/StyleTTS2/exp/332_128985_000001_000000.wav"

    if TEST_FIRST_MODEL:
        model_name = "mdit_cfm"  # "styletts2_txt2mel"  "mdit_cfm"
        first_model_path, first_config = model_root_dir + model_config[model_name][0], model_root_dir + model_config[model_name][1]
        first_model = get_first_model(ckpt=first_model_path, config_f=first_config)
        inference_first(text1, ref_wav1, first_model, out_wav_f="exp/res/mdit_first_syn.wav")

    if TEST_SECOND_MODEL:
        second_model, sampler, model_params = get_second_model(ckpt=second_model_path, config_f=second_config)
        inference_second(text1, ref_wav1, second_model, sampler, model_params,
                         alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1,
                         wav_n="exp/res/inference_1_or_2_2.wav")
    # test bunch of speech

    # test txt
    if TEST_TEXT:
        ### IN
        style = "exp/data/r1_50.txt"  # r1_50
        txt = "exp/data/s1_5.txt"  # s1_5
        dataset = "esd"  # libritts
        slice_num = 10

        ### OUT
        #out_dir = "/home/rosen/StableTTS/exp/styletts2/random_10"
        #out_dir = "/home/rosen/drawspeech/log/exp/lddpm_esd_basic/styletts2/random"

        syn_styles = get_synStyle_from_file(style, split_char='|', dataset_name=dataset)  # emotion changed
        synTexts = get_synText_from_file(txt)
        if slice_num > 0:
            syn_styles = syn_styles[:slice_num]
            synTexts = synTexts[:slice_num]

        # get model
        model_name = "styletts2_txt2mel"  # "styletts2_txt2mel"  "mdit_cfm"
        second_model_path, second_config = model_root_dir + model_config[model_name][2], model_root_dir + model_config[model_name][3]
        second_model, sampler, model_params = get_second_model(ckpt=second_model_path, config_f=second_config, model_name=model_name)

        # synthesized wav_dir (IN: model, style, txt, out)
        USE_STYLEDIFF = False
        USE_STYLEENC = True
        if USE_STYLEDIFF:
            glb_type = "stylediff"
            ref_psd_type = "ref_pe"
            epoch_n = second_model_path.split(".")[0][-2:]
            style_dim = 512
            out_dir = f"res/{model_name}_{glb_type}_{ref_psd_type}_epoch{epoch_n}"

            alpha = 1 if glb_type == "styleEnc" else 0.3  # weight of acoustic style using styleDiff
            beta = 1 if glb_type == "styleEnc" else 0.7   # weight of predicting style using styleDiff
            style_dim = 512 if "mdit_cfm" in model_name else 256
            syn_speech_by_second_model(synTexts, syn_styles, out_dir, second_model=second_model, sampler=sampler, model_params=model_params,
                                       pe_type=ref_psd_type, alpha=alpha, beta=beta, style_dim=style_dim)
        if USE_STYLEENC:
            glb_type = "stylediff"    #
            ref_psd_type = "ref_aware_pred_pe"  # Pred_pe, ref_aware_pred_pe, ref_pe
            epoch_n = second_model_path.split(".")[0][-2:]
            style_dim = 256
            out_dir = f"res/{model_name}_{glb_type}_{ref_psd_type}_epoch{epoch_n}"

            alpha = 1 if glb_type == "styleEnc" else 0.3  # weight of acoustic style using styleDiff
            beta = 1 if glb_type == "styleEnc" else 0.7   # weight of predicting style using styleDiff
            syn_speech_by_second_model(synTexts, syn_styles, out_dir, second_model=second_model, sampler=sampler,
                                       model_params=model_params, pe_type=ref_psd_type, alpha=alpha, beta=beta, style_dim=style_dim)