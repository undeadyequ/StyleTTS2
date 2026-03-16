import sys
import time

from exp.exec_draw_two_pitch import plot_pitch_multi

sys.path.append("../")
import os
import shutil
import yaml
import torch
import torchaudio
import librosa
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
from munch import Munch

os.chdir('/home/rosen/Project/StyleTTS2')

# Local imports (assumed to be in the same directory or PYTHONPATH)
from models import load_ASR_models, load_F0_models
from utils import recursive_munch, maximum_path, mask_from_lens, log_norm, append_sentence_to_file, length_to_mask
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import phonemizer
from Utils.PLBERT.util import load_plbert
from load_vocoder import get_vocoder
from exec_utmosv2 import run_utmos
from utils import frame_to_phoneme_avg_and_back_binary, align_dur2, get_cut_phonemes_by_cut_f2p_attn
import torch.nn.functional as F
from utils import extract_pitch_trend, extract_pitch_trend_v2
from exp_utils import build_voiced_mask
from exp_utils_bk import get_synText_from_file, get_synStyle_from_file

from exp.vis2 import plot_f0_comparison
from statcz_psd import calcualte_pitch_energy_dtw
from extract_psd import extract_psd_single

class InferenceAPI:
    def __init__(self, model_name, ckpt_path, config_path, vocoder_ckpt="/home/rosen/ckpt/styletts/Vocoder/LibriTTS/",
                 device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name

        # Load Config
        self.config = yaml.safe_load(open(config_path))
        self.model_params = recursive_munch(self.config['model_params'])

        # Build Text Tools
        self.text_cleaner = TextCleaner()
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True
        )

        # Build Audio Tools
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean, self.std = -4, 4

        # Load Modules
        self.vocoder = get_vocoder(ckpt_dir=vocoder_ckpt, device=self.device)

        # Load model
        self.model, self.sampler = self._load_model(ckpt_path)

        # Set to eval
        for key in self.model:
            self.model[key].to(self.device).eval()

    def _load_model(self, ckpt):
        # Load pretrained ASR, F0, and BERT as defined in config
        text_aligner = load_ASR_models(self.config.get('ASR_path'), self.config.get('ASR_config'))
        pitch_extractor = load_F0_models(self.config.get('F0_path'))
        plbert = load_plbert(self.config.get('PLBERT_dir'))

        if "styletts2_txt2mel" in self.model_name:
            from models_txt2mel import build_model
            model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        elif "mdit_cfm" in self.model_name or "decodit" in self.model_name:
            from models_txt2mel_cfm import build_model
            model = build_model(self.model_params, self.config['cfm_config'], text_aligner, pitch_extractor, plbert)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        # Load Weights
        params_whole = torch.load(ckpt, map_location='cpu')
        params = params_whole['net']
        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key])
                except:
                    # Handle DataParallel prefix
                    from collections import OrderedDict
                    new_state_dict = OrderedDict({k[7:]: v for k, v in params[key].items()})
                    model[key].load_state_dict(new_state_dict, strict=False)

        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )
        return model, sampler

    def _preprocess_wav(self, wav_path):
        wave, sr = librosa.load(wav_path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)

        wave_tensor = torch.from_numpy(audio).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        mel_tensor = mel_tensor.to(self.device)

        # Ensure even length
        mel_len = mel_tensor.size(-1)
        return mel_tensor[:, :, :(mel_len - mel_len % 2)]

    def _get_tokens(self, text):
        ps = self.global_phonemizer.phonemize([text.strip()])
        ps = word_tokenize(ps[0])
        tokens = self.text_cleaner(' '.join(ps))   #  tokens and ps may have different length due to unrecognized character!
        tokens.insert(0, 0)
        tokens.append(0)    # add this to consistence with training
        ps_append = ' ' + ' '.join(ps) + ' '
        #ps_append_len = len(ps_append)
        #print(len(tokens), ps_append_len)
        return ps_append, torch.LongTensor(tokens).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def synthesize_one(self, text, ref_wav_path, ref_txt, alpha=0.3, beta=0.7, diffusion_steps=5, cfg_strength=3.0,
                       hierStyle=True, drop_trend=False, return_pitch=False, trend_strength=1.0, need_uv_mask=True, return_trd_index=False,
                       return_attn_map=False):  # trend_strength/need_uv_mask
        """Synthesize a single speech sample."""
        ps, tokens = self._get_tokens(text)
        ps_ref, tokens_ref = self._get_tokens(ref_txt)
        ref_txt_lengths = torch.LongTensor([tokens_ref.shape[-1]]).to(self.device)

        ref_mel = self._preprocess_wav(ref_wav_path)

        # Compute Style
        with torch.no_grad():
            ref_s_enc = self.model.style_encoder(ref_mel.unsqueeze(1))
            ref_p_enc = self.model.predictor_encoder(ref_mel.unsqueeze(1))
            ref_s_combined = torch.cat([ref_s_enc, ref_p_enc], dim=1)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)
        output_lengths = torch.LongTensor([ref_mel.shape[-1]]).to(self.device)

        # Latent Encoding
        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        # Diffusion Style Sampling
        style_dim = 256
        s_pred = self.sampler(
            noise=torch.randn((1, style_dim)).unsqueeze(1).to(self.device),
            embedding=bert_dur,
            embedding_scale=1,
            features=ref_s_combined,
            num_steps=diffusion_steps
        ).squeeze(1)

        # Mix styles based on alpha/beta
        s = beta * s_pred[:, style_dim // 2:] + (1 - beta) * ref_s_combined[:, style_dim // 2:]
        ref = alpha * s_pred[:, :style_dim // 2] + (1 - alpha) * ref_s_combined[:, :style_dim // 2]

        # Duration & Prosody
        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.model.predictor.lstm(d)
        duration = torch.sigmoid(self.model.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        # Alignment
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data)).to(self.device)
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # Prosody conditioning
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0))
        asr = (t_en @ pred_aln_trg.unsqueeze(0))

        # Shift for decoder
        asr_new = torch.zeros_like(asr)
        asr_new[:, :, 1:] = asr[:, :, 0:-1]
        en_new = torch.zeros_like(en)
        en_new[:, :, 1:] = en[:, :, 0:-1]

        # F0_ref, s2s, uv_mask
        F0_ref, _, F0 = self.model.pitch_extractor(ref_mel.unsqueeze(1))
        N_real = log_norm(ref_mel.unsqueeze(1)).squeeze(1)
        s2s = get_s2s(tokens_ref, ref_txt_lengths, ref_mel, output_lengths, self.model.text_aligner, self.device)
        uv_masks = build_voiced_mask(list(ps)).unsqueeze(0).to(s2s.device)
        uv_ref_masks = build_voiced_mask(list(ps_ref)).unsqueeze(0).to(s2s.device)

        # Align uv masks to match s2s/pred_aln phoneme dims (text_cleaner may drop unknown chars)
        """
        s2s_phn_dim = s2s.size(1)
        if uv_ref_masks.size(-1) != s2s_phn_dim:
            uv_ref_masks = uv_ref_masks[:, :s2s_phn_dim] if uv_ref_masks.size(-1) > s2s_phn_dim \
                else F.pad(uv_ref_masks, (0, s2s_phn_dim - uv_ref_masks.size(-1)), "constant", 0)
        tgt_phn_dim = pred_aln_trg.size(0)  # target phoneme count
        if uv_masks.size(-1) != tgt_phn_dim:
            uv_masks = uv_masks[:, :tgt_phn_dim] if uv_masks.size(-1) > tgt_phn_dim \
                else F.pad(uv_masks, (0, tgt_phn_dim - uv_masks.size(-1)), "constant", 0)
        """

        if not need_uv_mask:
            uv_masks = torch.ones_like(uv_masks).to(uv_masks.device)
        if return_trd_index:
            trd, trd_index = self.model.predictor.trd_encoding(F0_ref, s2s, uv_ref_masks, uv_masks, pred_aln_trg.unsqueeze(0),
                                                               drop_trend=drop_trend, return_trd_index=return_trd_index)  # bert_trend_fused embeds
        else:
            trd = self.model.predictor.trd_encoding(F0_ref, s2s, uv_ref_masks, uv_masks, pred_aln_trg.unsqueeze(0), drop_trend=drop_trend,
                                                    return_trd_index=return_trd_index)  # bert_trend_fused embeds
        en_new = torch.cat([en_new, trd * trend_strength], dim=1)  #

        # get predicted pitch energy
        F0_pred, N_pred = self.model.predictor.F0Ntrain(en_new, s)
        pe = torch.cat([N_pred.unsqueeze(1), F0_pred.unsqueeze(1)], dim=1)

        # Decode Mel
        mel_rec, attn_maps = self.model.decoder(
            mu=asr_new, mask=None, n_timesteps=200, temperature=1.0,
            c=ref, seq_style=pe, p_mask=None, cfg_strength=cfg_strength,
            return_attn_map=return_attn_map)

        # Vocode
        audio = self.vocoder(mel_rec).squeeze().cpu().numpy()
        if return_attn_map:
            if return_pitch:
                if return_trd_index:
                    return audio, F0_pred, F0_ref, N_pred, N_real, trd_index.squeeze(), attn_maps
                return audio, F0_pred, F0_ref, N_pred, N_real, attn_maps
            return audio[..., :-50], attn_maps
        if return_pitch:
            if return_trd_index:
                return audio, F0_pred, F0_ref, N_pred, N_real, trd_index.squeeze()
            return audio, F0_pred, F0_ref, N_pred, N_real
        return audio[..., :-50]  # Trim boundary artifact

    def synthesize_batch(self, texts, ref_wav_path, ref_txt, out_dir, out_wav_prefix="syn", **kwargs):
        """Synthesize multiple samples. ref_wav_path should be a single path."""
        Path(out_dir).mkdir(exist_ok=True, parents=True)

        if isinstance(ref_wav_path, str):
            ref_wav_path = [ref_wav_path] * len(texts)
            ref_txts = [ref_txt] * len(texts)

        results = []
        for i, (text, ref_path, ref_txt) in enumerate(zip(texts, ref_wav_path, ref_txts)):
            audio = self.synthesize_one(text, ref_path, ref_txt, **kwargs)
            wav_f = os.path.join(out_dir, f"{out_wav_prefix}{i}.wav")
            txt_f = os.path.join(out_dir, f"{out_wav_prefix}{i}.lab")
            torchaudio.save(wav_f, torch.from_numpy(audio).unsqueeze(0), 24000)
            with open(txt_f, "w") as file1:
                file1.write(text)
            results.append(wav_f)
        return results


@torch.no_grad()
def get_s2s(texts, input_lengths, mels, mel_input_length, text_aligner, device="cuda"):
    """
    :param
    texts:  (B, T, D)
    mels:  (B, T, D) (T, D)
    text_aligner:
    """
    mask = length_to_mask(mel_input_length // 2).to(device)
    text_mask = length_to_mask(input_lengths).to(texts.device)
    _, _, s2s_attn = text_aligner(mels, mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)
    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // 2)
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    return s2s_attn_mono


if __name__ == '__main__':
    # Initialize API
    model_root_dir = "/home/rosen/ckpt/styletts2_libriTTS/"
    model_config = {
        "styletts2_txt2mel": ["first_txt2mel/epoch_1st_00048.pth", "first_txt2mel/config_libritts_txt2mel.yml",
                              "first_txt2mel/epoch_2nd_00028.pth", "first_txt2mel/config_libritts_txt2mel.yml"],
        "mdit_cfm_v10": ["", "",
                         "first_txt2mel_cfm_v10/epoch_2nd_00048.pth",
                         "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "mdit_cfm_v16": ["", "",
                         "first_txt2mel_cfm_v16/epoch_2nd_00020.pth",
                         "first_txt2mel_cfm_v16/config_libritts_txt2mel_cfm_v16.yml"],
        "mdit_cfm_v20": ["", "",
                         "first_txt2mel_cfm_v20/epoch_2nd_00018.pth",
                         "first_txt2mel_cfm_v20/config_libritts_txt2mel_cfm_v20.yml"],
        "decodit_cfm_v25": ["", "",
                         "first_txt2mel_cfm_v25/epoch_2nd_00036.pth",
                         "first_txt2mel_cfm_v25/config_libritts_txt2mel_cfm_v25_modify.yml"],
        "decodit_cfm_v26": ["", "",
                            "first_txt2mel_cfm_v26/epoch_2nd_00030.pth",
                            "first_txt2mel_cfm_v26/config_libritts_txt2mel_cfm_v26.yml"],
        "decodit_cfm_v27": ["", "",
                            "first_txt2mel_cfm_v27/epoch_2nd_00030.pth",
                            "first_txt2mel_cfm_v27/config_libritts_txt2mel_cfm_v27.yml"],
        "decodit_cfm_v28": ["", "",
                            "first_txt2mel_cfm_v28/epoch_2nd_00038.pth",
                            "first_txt2mel_cfm_v28/config_libritts_txt2mel_cfm_v28.yml"],
        "decodit_cfm_v29": ["", "",
                            "first_txt2mel_cfm_v29/epoch_2nd_00048.pth",
                            "first_txt2mel_cfm_v29/config_libritts_txt2mel_cfm_v29.yml"],
    }

    model_name = "decodit_cfm_v29"
    api = InferenceAPI(
        model_name=model_name,
        ckpt_path=model_root_dir + model_config[model_name][2],
        config_path=model_root_dir + model_config[model_name][3],
        device="cuda"
    )
    reference_wav = "/home/rosen/Project/StyleTTS2/exp/res/spk142824_2790_142824_000046_000000.wav_ref0_syn0.wav"
    reference_wav1 = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0019_Angry_ref0.wav"
    reference_wav1_lab = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0019_Angry_ref0.lab"

    reference_wav3 = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0019_Surprise_ref3.wav"
    reference_wav3_lab = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0019_Surprise_ref3.lab"

    #reference_wav3 = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0013_Angry_ref0.wav"
    #reference_wav3_lab = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0013_Angry_ref0.lab"

    text = "I must have two to fetch and carry."

    out_dir = "/home/rosen/Project/StyleTTS2/exp/res/v29"
    out_wav = f"{out_dir}/{model_name}.wav"

    if not os.path.isdir(os.path.dirname(out_wav)):
        os.makedirs(os.path.dirname(out_wav))

    # 1. Single Inference
    if True:
        hier_version = int(model_name[-2:])  # v[ver]
        hierStyle = True if hier_version > 15 else False

        # read ref txt
        with open(reference_wav3_lab) as f:
            reference_wav3_lab = f.read()

        # Base
        torch.manual_seed(0)
        out_wav_trend = f"{out_dir}/{model_name}.wav"  # _add10  _05strength
        wav_trd, F0_pred_trend, F0_ref, N_pred_trend, N_real, trd_index = api.synthesize_one(
            text, reference_wav3, reference_wav3_lab, hierStyle=hierStyle, drop_trend=False, return_pitch=True, cfg_strength=3.0, trend_strength=1.0, return_trd_index=True)
        torchaudio.save(out_wav_trend, torch.from_numpy(wav_trd).unsqueeze(0), sample_rate=24000)

        # Variant1: trend * 5.0
        variant1 = "stren2" # Stren5
        out_wav_trend_variant1 = f"{out_dir}/{model_name}_{variant1}.wav"  # strength: _05strength, uv:
        wav_trd_vairiant1, F0_pred_variant1, F0_ref, N_pred_variant1, N_real = api.synthesize_one(
            text, reference_wav3, reference_wav3_lab, hierStyle=hierStyle, drop_trend=False, return_pitch=True, cfg_strength=3.0, trend_strength=2.0,
            need_uv_mask=False)
        torchaudio.save(out_wav_trend_variant1, torch.from_numpy(wav_trd_vairiant1).unsqueeze(0), sample_rate=24000)

        # Variant2: noTrend -> add quantized data
        torch.manual_seed(0)
        variant2 = "notrend"
        out_wav_variant2 = f"{out_dir}/{model_name}_{variant2}.wav" # notrend
        wav_notrd, F0_pred_variant2, _, N_pred_variant2, _ = api.synthesize_one(
            text, reference_wav3, reference_wav3_lab, hierStyle=hierStyle, drop_trend=True, return_pitch=True, cfg_strength=3.0)
        torchaudio.save(out_wav_variant2, torch.from_numpy(wav_notrd).unsqueeze(0), sample_rate=24000)
        #uv_masks_frame = F.interpolate(uv_masks_frame.unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0)

        # plot difference of conditioning pitch due to different variatn
        plot_f0_comparison(
            contours=[F0_ref[0], F0_pred_trend[0].squeeze(0), F0_pred_variant1[0].squeeze(0)],
            labels=["pitch_ref", "pitch_base", f"pitch_{variant1}"],
            out_path=f"{out_dir}/{model_name}_pitch_ref_baseVS{variant1}.png")
        plot_f0_comparison(
            contours=[F0_ref[0], F0_pred_trend[0].squeeze(0), F0_pred_variant2[0], trd_index * 20],
            labels=["pitch_ref", "pitch_base", f"pitch_{variant2}", "trd_quant"],
            out_path=f"{out_dir}/{model_name}_pitch_ref_baseVS{variant2}.png")
        plot_pitch_multi(
            [reference_wav3, out_wav_trend, out_wav_variant2], ["reference", "base", variant2],
            sr=24000, hop_length=300,
            f0_floor=50, f0_ceil=600,
            normalize_time=True,  # set False if they’re same length
            out_pdf=f"{out_dir}/{model_name}_pitch_ref_baseVS{variant2}_synthesized.png")

        #print("F0_pred_notrend", F0_pred_notrend[0])
        #print("F0_pred_trend", F0_pred_trend[0].squeeze(0))
        plot_f0_comparison(
            contours=[N_real[0], N_pred_trend[0].squeeze(0), N_pred_variant1[0].squeeze(0)],
            labels=["energy_ref", "energy_base", f"energy_{variant1}"],
            out_path=f"{out_dir}/{model_name}_eng_ref_baseVS{variant1}.png")
        plot_f0_comparison(
            contours=[N_real[0], N_pred_trend[0].squeeze(0), N_pred_variant2[0]],
            labels=["energy_ref", "energy_base", f"energy_{variant2}"],
            out_path=f"{out_dir}/{model_name}_energy_ref_baseVS{variant2}.png")
        #plot_f0_comparison(uv_masks_frame.squeeze(0) * 100, F0_pred_notrend[0], F0_pred_trend[0].squeeze(0),
        #                   out_path=f"{out_dir}/{model_name}_pitch_uvmask_notrd_trd.png",
        #                   labels=("uvmask", "pitch_pred_notrend", "pitch_pred_trend"))  # test true/false
        #plot_pitch_multi([reference_wav3, out_wav_variant2, out_wav_trend, out_wav_trend_variant1], labels=["reference_wav3", "no_trd", "trd", "trd_stren5"],
        #                 out_pdf=f"{out_dir}/{model_name}_syn_pitch.png")

        #

        # calcuate dtw of notrend, trend, trendStren5
        """
        pith_ref, energy_ref = extract_psd_single(reference_wav3)
        pitch_variant1_syn, energy_variant1_syn = extract_psd_single(out_wav_trend_variant1)
        pitch_variant2_syn, energy_variatn2_syn = extract_psd_single(out_wav_variant2)
        pitch_trend_syn, energy_trend_syn = extract_psd_single(out_wav_trend)

        p_variant2_diff, e_variant2_diff = calcualte_pitch_energy_dtw(pith_ref, pitch_variant2_syn, energy_ref,
                                                                      energy_variatn2_syn, need_interp_unvoice=True)
        p_base_diff, e_base_diff = calcualte_pitch_energy_dtw(pith_ref, pitch_trend_syn, energy_ref,
                                                              energy_trend_syn, need_interp_unvoice=True)
        p_variant1_diff, e_variant1_diff = calcualte_pitch_energy_dtw(pith_ref, pitch_variant1_syn, energy_ref,
                                                                      energy_variant1_syn, need_interp_unvoice=True)
        """
        plot_f0_comparison(
            contours=[F0_ref[0], pitch_trend_syn, pitch_variant1_syn],
            labels=["pitch_ref", "pitch_base", f"pitch_{variant1}"],
            out_path=f"{out_dir}/{model_name}_pitch_ref_baseVS{variant1}.png")

        print("dtw pitch and energy of base: {}, {}: {}, and {}: {}".format(
            (p_base_diff, e_base_diff), variant1, (p_variant1_diff, e_variant1_diff), variant2, (p_variant2_diff, e_variant2_diff)))

    # 2. Batch Inference
    #texts = ["Your path now goes south."]

    # Test batch. Do experiment
    if False:
        start = time.time()
        dataset = "esd"  # libritts esd
        style = "exp/data/r1_50.txt"  # r1_50 r1_target libri_r1, r1_test, r1_sharp_last
        txt = "exp/data/s1.txt"  # s1_5 s1_target  libri_s1, s1_test, s1_10
        syn_styles = get_synStyle_from_file(style, split_char='|', dataset_name=dataset)  # emotion changed
        synTexts = get_synText_from_file(txt)

        ref_texts = []
        out_dir = "exp/res/parallel_test"
        for i, ref_s in enumerate(syn_styles):
            spk, emo, ref_txt, speech_path = ref_s
            ref_texts.append(ref_txt) if ref_txt not in ref_texts else ref_texts
            r_id = ref_texts.index(ref_txt)
            speech_id_1st = f'spk{spk}_{emo}_ref{r_id}_syn'
            api.synthesize_batch(synTexts, speech_path, ref_txt, out_dir=out_dir, out_wav_prefix=speech_id_1st,
                                 hierStyle=True, drop_trend=True)
        print("persistent workers synthesize taking time:", time.time() - start)

    # 120s, 263.28386926651; 250audio: multi_592, single_513, shared4_387, shared8_500
    # Save utmos_v2 score
    ##mean_mos, std_mos, mos_list = run_utmos(out_dir, 1)
    #print(f"\nAverage UTMOS-v2: {mean_mos:.4f} ± {std_mos:.4f}")
    #append_sentence_to_file("exp/utmosv2_log.txt", f"Average UTMOS-v2 of {out_dir}: {mean_mos:.4f} ± {std_mos:.4f}")
    # api.synthesize_batch(synTexts, speech_path, ref_txt, out_dir=out_dir, out_wav_prefix=speech_id_1st, hierStyle=True, drop_trend=True)

