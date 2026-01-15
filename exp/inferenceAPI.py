import sys
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
from utils import frame_to_phoneme_avg_and_back_binary, align_dur2
import torch.nn.functional as F


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
        elif "mdit_cfm" in self.model_name:
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
        tokens = self.text_cleaner(' '.join(ps))
        tokens.insert(0, 0)
        return torch.LongTensor(tokens).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def synthesize_one(self, text, ref_wav_path, alpha=0.3, beta=0.7, diffusion_steps=5, cfg_strength=None):
        """Synthesize a single speech sample."""
        tokens = self._get_tokens(text)
        ref_mel = self._preprocess_wav(ref_wav_path)

        # Compute Style
        with torch.no_grad():
            ref_s_enc = self.model.style_encoder(ref_mel.unsqueeze(1))
            ref_p_enc = self.model.predictor_encoder(ref_mel.unsqueeze(1))
            ref_s_combined = torch.cat([ref_s_enc, ref_p_enc], dim=1)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)

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

        F0_pred, N_pred = self.model.predictor.F0Ntrain(en_new, s)
        pe = torch.cat([N_pred.unsqueeze(1), F0_pred.unsqueeze(1)], dim=1)

        # Decode Mel
        mel_rec, _ = self.model.decoder(
            mu=asr_new, mask=None, n_timesteps=200, temperature=1.0,
            c=ref, seq_style=pe, p_mask=None, cfg_strength=cfg_strength, return_attn_map=False
        )

        # Vocode
        audio = self.vocoder(mel_rec).squeeze().cpu().numpy()
        return audio[..., :-50]  # Trim boundary artifact

    def synthesize_batch(self, texts, ref_wav_paths, out_dir, **kwargs):
        """Synthesize multiple samples. ref_wav_paths can be a list or a single path."""
        Path(out_dir).mkdir(exist_ok=True, parents=True)

        if isinstance(ref_wav_paths, str):
            ref_wav_paths = [ref_wav_paths] * len(texts)

        results = []
        for i, (text, ref_path) in enumerate(zip(texts, ref_wav_paths)):
            audio = self.synthesize_one(text, ref_path, **kwargs)
            out_path = os.path.join(out_dir, f"sample_{i}.wav")
            torchaudio.save(out_path, torch.from_numpy(audio).unsqueeze(0), 24000)
            results.append(out_path)

        return results

if __name__ == '__main__':
    # Initialize API
    api = InferenceAPI(
        model_name="mdit_cfm_v10",
        ckpt_path="/home/rosen/ckpt/styletts2_libriTTS/first_txt2mel_cfm_v10/epoch_2nd_00048.pth",
        config_path="/home/rosen/ckpt/styletts2_libriTTS/first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"
    )
    reference_wav = "/home/rosen/Project/StyleTTS2/exp/res/spk142824_2790_142824_000046_000000.wav_ref0_syn0.wav"
    out_wav = "/home/rosen/Project/StyleTTS2/exp/res/out_1.wav"

    # 1. Single Inference
    #wav = api.synthesize_one("Hello, this is a test of the reorganized API.", reference_wav)
    #torchaudio.save(out_wav, torch.from_numpy(wav).unsqueeze(0), sample_rate=24000)

    # 2. Batch Inference
    texts = ["First sentence.", "Second sentence."]

    api.synthesize_batch(texts, reference_wav, out_dir="exp/res")