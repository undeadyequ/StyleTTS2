"""inferenceAPI_bertFusion_icl.py

Inference API for CFMDecoderV4 (F5-TTS-style in-context learning).

Key differences from inferenceAPI_bertFusion.py:
  - Uses model.decoder(mu_tgt, cond_ref, mu_ref, seq_style_tgt, seq_style_ref, cfg_strength=...)
    instead of model.decoder(mu=..., c=ref, seq_style=pe, ...).
  - style_encoder is NOT used for decoder conditioning (no global c).
    style_encoder + predictor_encoder are used for diffusion-based prosodic style sampling.
  - mu_ref is the frame-level reference text encoding computed via forced alignment between
    ref_mel and ref_text:  t_en_ref @ s2s_attn_mono_ref  (same as 'asr' in training).
  - ref_mel is optionally sliced to max_ref_mel_frames; ref_text phonemes are adaptively
    trimmed to match the sliced duration (using per-phoneme frame counts from the alignment).
"""

import sys
sys.path.append("../")
import os
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

from models import load_ASR_models, load_F0_models
from utils import recursive_munch, maximum_path, mask_from_lens, log_norm, length_to_mask
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import phonemizer
from Utils.PLBERT.util import load_plbert
from load_vocoder import get_vocoder
import torch.nn.functional as F
from exp_utils import build_voiced_mask


class InferenceAPIICL:
    """Inference API for CFMDecoderV4 with F5-TTS style in-context learning.

    Uses forced alignment between ref_mel and ref_text to produce the frame-aligned
    reference text encoding (mu_ref) that the v4 decoder requires.  The reference mel
    can optionally be clipped to a maximum length, with ref_text phonemes trimmed
    adaptively based on phoneme-to-frame durations from the alignment.
    """

    def __init__(self, model_name, ckpt_path, config_path,
                 vocoder_ckpt="/home/rosen/ckpt/styletts/Vocoder/LibriTTS/",
                 device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name

        self.config = yaml.safe_load(open(config_path))
        self.model_params = recursive_munch(self.config['model_params'])

        self.text_cleaner = TextCleaner()
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True
        )

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean, self.std = -4, 4

        self.vocoder = get_vocoder(ckpt_dir=vocoder_ckpt, device=self.device)
        self.model, self.sampler = self._load_model(ckpt_path)
        for key in self.model:
            self.model[key].to(self.device).eval()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, ckpt):
        text_aligner = load_ASR_models(self.config.get('ASR_path'), self.config.get('ASR_config'))
        pitch_extractor = load_F0_models(self.config.get('F0_path'))
        plbert = load_plbert(self.config.get('PLBERT_dir'))

        from models_txt2mel_cfm import build_model_v4
        model = build_model_v4(self.model_params, self.config['cfm_config'],
                               text_aligner, pitch_extractor, plbert)

        params_whole = torch.load(ckpt, map_location='cpu')
        params = params_whole['net']
        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key])
                except Exception:
                    from collections import OrderedDict
                    new_state = OrderedDict({k[7:]: v for k, v in params[key].items()})
                    model[key].load_state_dict(new_state, strict=False)

        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )
        return model, sampler

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_wav(self, wav_path):
        """Load, trim, compute log-mel, ensure even length."""
        wave, sr = librosa.load(wav_path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        wave_tensor = torch.from_numpy(audio).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        mel_tensor = mel_tensor.to(self.device)
        mel_len = mel_tensor.size(-1)
        return mel_tensor[:, :, :(mel_len - mel_len % 2)]  # [1, 80, T_even]

    def _get_tokens(self, text):
        """Phonemise text and convert to token ids."""
        ps = self.global_phonemizer.phonemize([text.strip()])
        ps = word_tokenize(ps[0])
        tokens = self.text_cleaner(' '.join(ps))
        tokens.insert(0, 0)
        tokens.append(0)
        ps_str = ' ' + ' '.join(ps) + ' '
        return ps_str, torch.LongTensor(tokens).to(self.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Main synthesis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def synthesize_one(self, text, ref_wav_path, ref_txt,
                       max_ref_mel_frames=None, alpha=0.3, beta=0.7,
                       diffusion_steps=5, cfg_strength=3.0,
                       drop_trend=False, trend_strength=1.0, need_uv_mask=True,
                       return_attn_map=False, return_pitch=False):
        """Synthesize speech with F5-TTS style in-context learning.

        Args:
            text               : target text to synthesise.
            ref_wav_path       : path to reference audio (speaker / style source).
            ref_txt            : transcript of the reference audio.
            max_ref_mel_frames : clip reference mel to at most this many frames
                                 (None = use full reference).
            alpha              : unused in v4; kept for API compatibility.
            beta               : prosodic style mixing weight (diffusion vs. ref).
            diffusion_steps    : number of diffusion sampling steps.
            cfg_strength       : CFG scale for the CFMDecoderV4 (None = no CFG).
            drop_trend         : if True, zero the pitch-trend conditioning.
            trend_strength     : scale applied to the trend embedding.
            need_uv_mask       : if False, treat all target phones as voiced.
            return_attn_map    : also return decoder cross-attention maps.
            return_pitch       : also return F0 / energy predictions.

        Returns:
            audio    : numpy waveform at 24 kHz.
            (optional) F0_pred, F0_ref, N_pred, N_ref, attn_maps
        """
        # -- 1. Tokenise target & reference texts --
        ps, tokens = self._get_tokens(text)
        ps_ref, tokens_ref = self._get_tokens(ref_txt)

        input_lengths    = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        ref_txt_lengths  = torch.LongTensor([tokens_ref.shape[-1]]).to(self.device)
        text_mask        = length_to_mask(input_lengths).to(self.device)
        ref_txt_mask     = length_to_mask(ref_txt_lengths).to(self.device)

        # -- 2. Load reference audio → log-mel --
        ref_mel = self._preprocess_wav(ref_wav_path)   # [1, 80, T_ref_full]

        # -- 3. Optionally clip reference mel to max_ref_mel_frames --
        if max_ref_mel_frames is not None and ref_mel.size(-1) > max_ref_mel_frames:
            T_ref = (min(max_ref_mel_frames, ref_mel.size(-1)) // 2) * 2  # keep even
            ref_mel_sliced = ref_mel[:, :, :T_ref]
        else:
            ref_mel_sliced = ref_mel
        T_ref     = ref_mel_sliced.size(-1)   # mel frames in (possibly sliced) ref
        T_ref_asr = T_ref // 2                # ASR (half-rate) frames

        # -- 4. Force-align ref_text to FULL ref_mel --
        # Compute on full mel for better alignment quality, then slice afterwards.
        ref_output_lengths = torch.LongTensor([ref_mel.size(-1)]).to(self.device)
        s2s_attn_mono_ref = _get_s2s(
            tokens_ref, ref_txt_lengths, ref_mel, ref_output_lengths,
            self.model.text_aligner, self.device)
        # s2s_attn_mono_ref: [1, n_ref_phn, T_ref_full_asr]

        # -- 5. Frame-level reference text encoding  (mu_ref for decoder) --
        t_en_ref   = self.model.text_encoder(tokens_ref, ref_txt_lengths, ref_txt_mask)
        en_ref_full = t_en_ref @ s2s_attn_mono_ref         # [1, dim, T_ref_full_asr]
        en_ref      = en_ref_full[:, :, :T_ref_asr]        # [1, dim, T_ref_asr]

        # -- 6. Voiced mask for ref phonemes (full reference, for trd_encoding) --
        n_ref_phn    = s2s_attn_mono_ref.size(1)
        uv_ref_masks = build_voiced_mask(list(ps_ref)).unsqueeze(0).to(self.device)

        # Clamp/pad to match n_ref_phn (text_cleaner may drop unknown chars)
        if uv_ref_masks.size(-1) > n_ref_phn:
            uv_ref_masks = uv_ref_masks[:, :n_ref_phn]
        elif uv_ref_masks.size(-1) < n_ref_phn:
            uv_ref_masks = F.pad(uv_ref_masks, (0, n_ref_phn - uv_ref_masks.size(-1)))

        # -- 7. Pitch / energy from full reference mel (for trd_encoding prosody transfer) --
        F0_ref, _, _ = self.model.pitch_extractor(ref_mel.unsqueeze(1))

        # -- 7b. Pitch / energy from sliced reference mel (for decoder seq_style_ref) --
        F0_ref_sliced, _, _ = self.model.pitch_extractor(ref_mel_sliced.unsqueeze(1))
        norm_ref = log_norm(ref_mel_sliced.unsqueeze(1)).squeeze(1)
        pe_ref   = torch.cat([norm_ref.unsqueeze(1), F0_ref_sliced.unsqueeze(1)], dim=1)
        # pe_ref: [1, 2, T_ref]  — temporal conditioning for the ICL ODE ref portion

        # == Target TTS synthesis ==

        # -- 8. Target text encoding --
        t_en     = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en     = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        # -- 9. Diffusion-based prosodic style (conditioned on full ref_mel) --
        # ref_sp is used for duration prediction and global prosodic style;
        # extracted from the FULL reference mel so it captures the complete utterance.
        # ref_ss (acoustic style) is not used — speaker identity is supplied to the
        # decoder via cond_ref (the sliced mel), so it is not needed here.
        style_dim = self.model_params.style_dim   # 128
        ref_sp    = self.model.predictor_encoder(ref_mel.unsqueeze(1))  # [1, 128]
        s_pred = self.sampler(
            noise=torch.randn((1, style_dim)).unsqueeze(1).to(self.device),
            embedding=bert_dur,
            embedding_scale=1,
            features=ref_sp,
            num_steps=diffusion_steps
        ).squeeze(1)  # [1, style_dim]

        # Prosodic style for predictor: mix diffusion prediction with reference
        s = beta * s_pred + (1 - beta) * ref_sp   # [1, 128]

        # -- 10. Duration prediction → phoneme-to-frame alignment --
        d        = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _     = self.model.predictor.lstm(d)
        duration = torch.sigmoid(self.model.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)  # single-phoneme edge case

        n_phn      = input_lengths.item()
        T_tgt_asr  = int(pred_dur.sum().data)
        pred_aln_trg = torch.zeros(n_phn, T_tgt_asr).to(self.device)
        c_frame = 0
        for i in range(n_phn):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # -- 11. Frame-level embeddings for decoder (mu_tgt) and predictor (en_tts) --
        mu_tgt = t_en @ pred_aln_trg.unsqueeze(0)               # [1, dim, T_tgt_asr]
        sem_style = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)  # [1, dim, T_tgt_asr]

        # -- 12. Bert-fusion sem_pros encoding --
        uv_masks = build_voiced_mask(list(ps)).unsqueeze(0).to(self.device)
        # Clamp/pad uv_masks to n_phn (text_cleaner may drop unknown chars)
        if uv_masks.size(-1) > n_phn:
            uv_masks = uv_masks[:, :n_phn]
        elif uv_masks.size(-1) < n_phn:
            uv_masks = F.pad(uv_masks, (0, n_phn - uv_masks.size(-1)))
        if not need_uv_mask:
            uv_masks = torch.ones_like(uv_masks)
        pros_style = self.model.predictor.trd_encoding(
            F0_ref,               # pitch from FULL reference mel
            s2s_attn_mono_ref,    # full phoneme-to-frame alignment
            uv_ref_masks,         # full ref voiced mask
            uv_masks,
            pred_aln_trg.unsqueeze(0),
            drop_trend=drop_trend,
            return_trd_index=False
        )
        sem_pros_style = torch.cat([sem_style, pros_style * trend_strength], dim=1)  #

        # -- 13. Predict target pitch / energy --
        F0_pred, N_pred = self.model.predictor.F0Ntrain(sem_pros_style, s)
        pe_tgt = torch.cat([N_pred.unsqueeze(1), F0_pred.unsqueeze(1)], dim=1)
        # pe_tgt: [1, 2, T_tgt_mel]  (F0Ntrain upsamples ASR → mel internally)

        # -- 14. CFMDecoderV4 ICL inference --
        mel_rec, attn_maps = self.model.decoder(
            mu_tgt=mu_tgt,
            n_timesteps=200,
            temperature=1.0,
            cond_ref=ref_mel_sliced,
            mu_ref=en_ref,
            seq_style_tgt=pe_tgt,
            seq_style_ref=pe_ref,
            cfg_strength=cfg_strength,
            return_attn_map=return_attn_map
        )

        # -- 15. Vocode --
        audio = self.vocoder(mel_rec).squeeze().cpu().numpy()

        if return_attn_map:
            if return_pitch:
                return audio[..., :-50], F0_pred, F0_ref, N_pred, norm_ref, attn_maps
            return audio[..., :-50], attn_maps
        if return_pitch:
            return audio[..., :-50], F0_pred, F0_ref, N_pred, norm_ref
        return audio[..., :-50]

    # ------------------------------------------------------------------
    # Batch synthesis
    # ------------------------------------------------------------------

    def synthesize_batch(self, texts, ref_wav_path, ref_txt, out_dir,
                         out_wav_prefix="syn", **kwargs):
        """Synthesize multiple utterances sharing the same reference."""
        Path(out_dir).mkdir(exist_ok=True, parents=True)

        if isinstance(ref_wav_path, str):
            ref_wav_paths = [ref_wav_path] * len(texts)
            ref_txts      = [ref_txt]      * len(texts)
        else:
            ref_wav_paths = ref_wav_path
            ref_txts      = ref_txt

        results = []
        for i, (text, ref_path, ref_txt_item) in enumerate(
                zip(texts, ref_wav_paths, ref_txts)):
            audio = self.synthesize_one(text, ref_path, ref_txt_item, **kwargs)
            wav_f = os.path.join(out_dir, f"{out_wav_prefix}{i}.wav")
            txt_f = os.path.join(out_dir, f"{out_wav_prefix}{i}.lab")
            torchaudio.save(wav_f, torch.from_numpy(audio).unsqueeze(0), 24000)
            with open(txt_f, "w") as f:
                f.write(text)
            results.append(wav_f)
        return results


# ------------------------------------------------------------------
# Force-alignment helper
# ------------------------------------------------------------------

@torch.no_grad()
def _get_s2s(texts, input_lengths, mels, mel_input_length, text_aligner, device="cuda"):
    """Compute monotonic phoneme-to-ASR-frame attention via forced alignment.

    Returns:
        s2s_attn_mono : [B, n_phn, T_asr]  where T_asr = mel_input_length // 2
    """
    mask      = length_to_mask(mel_input_length // 2).to(device)
    text_mask = length_to_mask(input_lengths).to(texts.device)
    _, _, s2s_attn = text_aligner(mels, mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)
    mask_ST   = mask_from_lens(s2s_attn, input_lengths, mel_input_length // 2)
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    return s2s_attn_mono


# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------

if __name__ == '__main__':
    model_root_dir = "/home/rosen/ckpt/styletts2_libriTTS/"
    model_config = {
        "decodit_cfm_v36": [
            "first_txt2mel_cfm_v36/epoch_2nd_00016.pth",
            "first_txt2mel_cfm_v36/config_libritts_txt2mel_cfm_v36.yml",
        ],
    }

    model_name = "decodit_cfm_v36"
    cfg_path   = model_root_dir + model_config[model_name][1]
    ckpt_path  = model_root_dir + model_config[model_name][0]

    # Determine CFG strength from config
    cfm_cfg        = yaml.safe_load(open(cfg_path)).get('cfm_config', {})
    cfg_strength_v = 3.0 if cfm_cfg.get('cfg_dropout', 0) > 0 else None

    api = InferenceAPIICL(
        model_name=model_name,
        ckpt_path=ckpt_path,
        config_path=cfg_path,
        device="cuda"
    )

    reference_wav = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0019_Angry_ref0.wav"
    reference_lab = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0019_Angry_ref0.lab"
    text          = "I must have two to fetch and carry."

    with open(reference_lab) as f:
        ref_txt = f.read().strip()

    out_dir = "/home/rosen/Project/StyleTTS2/exp/res/v36"
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(0)
    audio = api.synthesize_one(
        text, reference_wav, ref_txt,
        max_ref_mel_frames=300,
        cfg_strength=cfg_strength_v,
        drop_trend=False,
        trend_strength=1.0,
    )
    out_wav = f"{out_dir}/{model_name}.wav"
    torchaudio.save(out_wav, torch.from_numpy(audio).unsqueeze(0), sample_rate=24000)
    print(f"Saved: {out_wav}")