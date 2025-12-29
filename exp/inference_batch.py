import time
from itertools import accumulate
from typing import List, Union, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
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

def _pad_1d_long(seqs: List[torch.LongTensor], pad_value: int = 0) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    seqs: list of [L] LongTensor on device
    returns:
      padded: [B, Lmax]
      lengths: [B]
    """
    lengths = torch.LongTensor([s.numel() for s in seqs]).to(seqs[0].device)
    Lmax = int(lengths.max().item())
    out = seqs[0].new_full((len(seqs), Lmax), pad_value)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out, lengths


def _build_pred_aln_trg_from_dur(pred_dur_1d: torch.Tensor) -> torch.Tensor:
    """
    pred_dur_1d: [L_text] (int-like), >=1
    return: [L_text, T] with 0/1
    """
    L = int(pred_dur_1d.numel())
    T = int(pred_dur_1d.sum().item())
    aln = pred_dur_1d.new_zeros((L, T), dtype=torch.float32)
    c = 0
    for i in range(L):
        d = int(pred_dur_1d[i].item())
        aln[i, c : c + d] = 1.0
        c += d
    return aln


def _shift_right_1frame(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, T]
    returns: [B, C, T] where frame t uses previous frame (ASR shift trick in your code)
    """
    x_new = torch.zeros_like(x)
    x_new[..., 0] = x[..., 0]
    x_new[..., 1:] = x[..., :-1]
    return x_new


@torch.no_grad()
def inference_batch(
    texts: List[str],
    ref_wavs: Union[List[Any], Any],  # whatever your extract_mel_from_wav / compute_style expects
    model,
    sampler,
    model_params,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
    style_dim: int = 256,
    mix_ref_pe_type: str = "none",
    cfg_strength: Optional[float] = None,
    mono_guide_delta: float = 0.0,
    fuse_beta: float = 0.3,
    # speed controls
    sort_by_text_len: bool = True,
) -> Dict[str, Any]:
    """
    Batched version of your inference_second.

    Key idea:
      - Batch what is naturally batchable: text encoder, BERT, diffusion style sampler.
      - Duration/alignment + mel/pitch extraction is variable-length; we handle per-sample,
        then pad to max acoustic length and batch the downstream predictor + decoder.

    Returns dict:
      {
        "wavs": List[np.ndarray],
        "pe": {
          "F0_ref": List[Tensor], "F0_pred": List[Tensor], "F0_cond": List[Tensor],
          "N_ref":  List[Tensor], "N_pred":  List[Tensor], "N_cond":  List[Tensor],
        },
        "attn_maps": List[Any],
        "pred_dur": List[Tensor],
        "order": List[int],  # mapping back to input order
      }
    """
    device = next(model.parameters()).device
    t0 = time.time()

    B = len(texts)
    assert B > 0, "texts is empty"

    # allow passing one reference for all texts
    if not isinstance(ref_wavs, (list, tuple)):
        ref_wavs = [ref_wavs] * B
    assert len(ref_wavs) == B, "len(ref_wavs) must match len(texts)"

    # (optional) sort by text length to reduce padding
    order = list(range(B))
    if sort_by_text_len:
        order = sorted(order, key=lambda i: len(texts[i]))
        texts = [texts[i] for i in order]
        ref_wavs = [ref_wavs[i] for i in order]

    # -----------------------------
    # 1) Text preprocessing (CPU-ish)
    # -----------------------------
    global_phonemizer, textclenaer = get_pretrained_modules()
    token_seqs = []
    ps_strings = []  # keep per-sample phoneme string/list for voiced mask logic
    for text in texts:
        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        ps_strings.append(ps)

        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        token_seqs.append(torch.LongTensor(tokens).to(device))

    tokens_padded, input_lengths = _pad_1d_long([t for t in token_seqs], pad_value=0)  # [B, Lmax]
    text_mask = length_to_mask(input_lengths).to(device)  # [B, Lmax] (True for padded, as in your code)

    # -----------------------------
    # 2) Ref preprocessing (variable length => do per-sample)
    # -----------------------------
    ref_mels = []
    ref_s_list = []
    for w in ref_wavs:
        ref_mel = extract_mel_from_wav(w)  # expected [n_mels, T] (your code uses .unsqueeze(1) later)
        ref_s = compute_style(w, model)    # expected [1, style_dim] (?) in your code
        ref_mels.append(ref_mel)
        ref_s_list.append(ref_s)

    # stack ref styles => [B, style_dim]
    ref_s = torch.cat(ref_s_list, dim=0).to(device)

    # -----------------------------
    # 3) Encode text + BERT (batch)
    # -----------------------------
    t_en = model.text_encoder(tokens_padded, input_lengths, text_mask)  # [B, C_txt, Lmax] (assumed)
    bert_dur = model.bert(tokens_padded, attention_mask=(~text_mask).int())  # [B, Lmax, H]
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)  # [B, C_bert, Lmax]

    # -----------------------------
    # 4) Diffusion style sampling (batch)
    # -----------------------------
    noise = torch.randn((B, style_dim), device=device).unsqueeze(1)  # [B, 1, style_dim]
    s_pred = sampler(
        noise=noise,
        embedding=bert_dur,
        embedding_scale=embedding_scale,
        features=ref_s,
        num_steps=diffusion_steps
    ).squeeze(1)  # [B, style_dim]

    # Handle NaN per-sample (your code checks only [0,0])
    half = style_dim // 2
    nan_mask = torch.isnan(s_pred[:, 0])

    # default: mix diffusion + ref encoding
    s = beta * s_pred[:, half:] + (1 - beta) * ref_s[:, half:]      # [B, half]
    ref = alpha * s_pred[:, :half] + (1 - alpha) * ref_s[:, :half]  # [B, half]

    # if NaN => fallback exactly like your code
    if nan_mask.any():
        s[nan_mask] = ref_s[nan_mask, half:]
        ref[nan_mask] = ref_s[nan_mask, :half]

    # -----------------------------
    # 5) Duration prediction (batch up to proj, then per-sample alignment)
    # -----------------------------
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)  # [B, C, Lmax]
    x, _ = model.predictor.lstm(d)                                       # [B, Lmax, H] (assumed)
    duration = model.predictor.duration_proj(x)                          # [B, Lmax, K?]
    duration = torch.sigmoid(duration).sum(axis=-1)                      # [B, Lmax]

    pred_dur_list = []
    pred_aln_list = []
    T_list = []

    for b in range(B):
        Lb = int(input_lengths[b].item())
        pred_dur = torch.round(duration[b, :Lb]).clamp(min=1).to(torch.long)  # [Lb]
        pred_dur_list.append(pred_dur)
        aln = _build_pred_aln_trg_from_dur(pred_dur).to(device)               # [Lb, Tb]
        pred_aln_list.append(aln)
        T_list.append(aln.size(1))

    Tmax = int(max(T_list))

    # pad alignment to [B, Lmax, Tmax]
    pred_aln_trg = torch.zeros((B, tokens_padded.size(1), Tmax), device=device, dtype=torch.float32)
    for b in range(B):
        Lb = int(input_lengths[b].item())
        Tb = pred_aln_list[b].size(1)
        pred_aln_trg[b, :Lb, :Tb] = pred_aln_list[b]

    # -----------------------------
    # 6) PE prediction (batch)
    # -----------------------------
    # en = d^T @ aln  => [B, C, Tmax]
    en = torch.bmm(d.transpose(-1, -2), pred_aln_trg)  # [B, C, Tmax]
    en = _shift_right_1frame(en)

    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)  # expected [B, Tmax*2?] depending on your setup

    # Ref pitch/energy extraction is variable-length => per-sample
    F0_ref_list, N_ref_list = [], []
    for b in range(B):
        ref_mel = ref_mels[b].to(device)  # [n_mels, T]
        F0_ref, _, _ = model.pitch_extractor(ref_mel.unsqueeze(0).unsqueeze(1))  # match your ref_mel.unsqueeze(1)
        N_ref = log_norm(ref_mel.unsqueeze(0).unsqueeze(1)).squeeze(1).detach()
        F0_ref_list.append(F0_ref.squeeze(0))  # [T_ref] or [1, T_ref] depending on extractor
        N_ref_list.append(N_ref.squeeze(0))

    # Build F0_cond/N_cond per-sample (since some branches need voiced mask)
    F0_cond_list, N_cond_list = [], []
    for b in range(B):
        if mix_ref_pe_type == "none":
            F0c, Nc = F0_pred[b], N_pred[b]
        elif mix_ref_pe_type == "ref_pe":
            F0c, Nc = F0_ref_list[b].squeeze(), N_ref_list[b].squeeze()
        elif mix_ref_pe_type in ("ref_pred_gate", "ref_pred_add"):
            ps = ps_strings[b]
            ps_list = [p for p in ps]
            ps_list.insert(0, " ")
            voiced_mask = build_voiced_mask(ps_list, pred_aln_list[b])  # uses unpadded [Lb, Tb]
            voiced_mask_upsampled = F.interpolate(
                voiced_mask.unsqueeze(0).unsqueeze(0),
                scale_factor=2,
                mode="nearest"
            ).squeeze()

            if mix_ref_pe_type == "ref_pred_gate":
                F0c, Nc, _a = fuse_prosody_final(
                    F0_pred[b].squeeze(), N_pred[b].squeeze(),
                    F0_ref_list[b].squeeze(), N_ref_list[b].squeeze(),
                    voiced_mask_upsampled, tau=0.15
                )
            else:
                F0c, Nc, _a = fuse_prosody_smooth_additive(
                    F0_pred[b].squeeze(), N_pred[b].squeeze(),
                    F0_ref_list[b].squeeze(), N_ref_list[b].squeeze(),
                    voiced_mask_upsampled, tau=0.15, beta=fuse_beta
                )
        else:
            raise ValueError(f"mix_ref_pe_type not supported: {mix_ref_pe_type}")

        # ensure 1D tensors on device
        F0_cond_list.append(F0c.to(device).flatten())
        N_cond_list.append(Nc.to(device).flatten())

    # pad F0/N condition to a common length for decoder batching
    # (your decoder expects ~2x frames relative to asr in hifigan path)
    # We'll compute asr first, then resize each F0/N to 2*T_asr(b).
    # -----------------------------
    # 7) txt emb prediction (batch)
    # -----------------------------
    asr = torch.bmm(t_en, pred_aln_trg)  # [B, C_txt, Tmax]
    asr = _shift_right_1frame(asr)

    # -----------------------------
    # 8) Decode (batched where possible)
    # -----------------------------
    attn_maps_out = [None] * B

    if model_params.decoder.type == "hifigan":
        # HifiGAN path: decode per-sample because mel length must match exactly and generator is usually per-sample anyway
        wavs = []
        for b in range(B):
            Tb = T_list[b]
            asr_b = asr[b:b+1, :, :Tb]  # [1, C, Tb]

            target_len = asr_b.size(-1) * 2
            F0b = F0_cond_list[b].unsqueeze(0).unsqueeze(0)  # [1,1,T?]
            Nb  = N_cond_list[b].unsqueeze(0).unsqueeze(0)

            if F0b.size(-1) != target_len:
                F0b = F.interpolate(F0b, size=target_len, mode="linear", align_corners=True)
                Nb  = F.interpolate(Nb,  size=target_len, mode="linear", align_corners=True)

            mel_rec = model.decoder(
                asr_b,
                F0b.squeeze(1),  # [1, target_len]
                Nb.squeeze(1),   # [1, target_len]
                ref[b:b+1].unsqueeze(0) if ref.dim() == 2 else ref[b:b+1],  # be tolerant
            )
            c = mel_rec.squeeze(0)                # [n_mels, Tm]
            out = generator(c.unsqueeze(0))       # your generator
            out = out.squeeze().cpu().numpy()[..., :-50]
            wavs.append(out)

    elif model_params.decoder.type == "mdit_cfm":
        # Batch-friendly path: pad seq_style to [B, 2, 2*Tmax] and provide mask if your decoder uses it.
        # pe: cat([N, F0]) along channel dim
        # First, ensure each sample has length 2*Tb (or 2*Tmax after padding) by interpolation.
        pe_list = []
        for b in range(B):
            Tb = T_list[b]
            target_len = Tb * 2

            F0b = F0_cond_list[b].unsqueeze(0).unsqueeze(0)  # [1,1,L]
            Nb  = N_cond_list[b].unsqueeze(0).unsqueeze(0)

            if F0b.size(-1) != target_len:
                F0b = F.interpolate(F0b, size=target_len, mode="linear", align_corners=True)
                Nb  = F.interpolate(Nb,  size=target_len, mode="linear", align_corners=True)

            pe_b = torch.cat([Nb, F0b], dim=1).squeeze(0)  # [2, target_len]
            pe_list.append(pe_b)

        pe_max = max(p.size(-1) for p in pe_list)
        pe = torch.zeros((B, 2, pe_max), device=device, dtype=pe_list[0].dtype)
        pe_mask = torch.ones((B, pe_max), device=device, dtype=torch.bool)  # True = padded
        for b in range(B):
            L = pe_list[b].size(-1)
            pe[b, :, :L] = pe_list[b]
            pe_mask[b, :L] = False

        # also pad mu/asr to Tmax (already) but create a mu_mask for valid frames Tb
        mu = asr  # [B, C, Tmax]
        mu_mask = torch.ones((B, Tmax), device=device, dtype=torch.bool)
        for b in range(B):
            mu_mask[b, :T_list[b]] = False

        mel_rec, attn_maps = model.decoder(
            mu=mu,
            mask=mu_mask,                 # if your decoder expects None, set None; but mask usually helps with padding
            n_timesteps=200,
            temperature=1.0,
            c=ref,                        # [B, half]
            seq_style=pe,                 # [B, 2, pe_max]
            p_mask=pe_mask,               # if your decoder uses it; otherwise set None
            cfg_strength=cfg_strength,
            mono_guide_delta=mono_guide_delta,
        )
        # split outputs per-sample
        wavs = []
        for b in range(B):
            # mel_rec likely [B, n_mels, Tm_max]; trim using something consistent with your implementation if needed
            c = mel_rec[b]
            out = generator(c.unsqueeze(0))
            out = out.squeeze().cpu().numpy()[..., :-50]
            wavs.append(out)
            if isinstance(attn_maps, (list, tuple)):
                attn_maps_out[b] = attn_maps[b] if len(attn_maps) == B else attn_maps
            else:
                attn_maps_out[b] = attn_maps

    else:
        raise ValueError(f"decoder type not supported: {model_params.decoder.type}")

    # -----------------------------
    # 9) Pack outputs (map back to original order)
    # -----------------------------
    # Gather PE tensors (keep as torch tensors; caller can move/convert)
    pe_out = {
        "F0_ref": [F0_ref_list[b].detach() for b in range(B)],
        "F0_pred": [F0_pred[b].detach() for b in range(B)],
        "F0_cond": [F0_cond_list[b].detach() for b in range(B)],
        "N_ref": [N_ref_list[b].detach() for b in range(B)],
        "N_pred": [N_pred[b].detach() for b in range(B)],
        "N_cond": [N_cond_list[b].detach() for b in range(B)],
    }

    # unsort to original
    inv = [0] * B
    for new_i, old_i in enumerate(order):
        inv[old_i] = new_i

    wavs_unsorted = [None] * B
    attn_unsorted = [None] * B
    pred_dur_unsorted = [None] * B
    pe_unsorted = {k: [None] * B for k in pe_out.keys()}

    for old_i in range(B):
        new_i = inv[old_i]
        wavs_unsorted[old_i] = wavs[new_i]
        attn_unsorted[old_i] = attn_maps_out[new_i]
        pred_dur_unsorted[old_i] = pred_dur_list[new_i].detach()
        for k in pe_out.keys():
            pe_unsorted[k][old_i] = pe_out[k][new_i]

    print(f"[inference_batch] B={B} done in {time.time()-t0:.3f}s (decoder={model_params.decoder.type})")

    return {
        "wavs": wavs_unsorted,
        "pe": pe_unsorted,
        "attn_maps": attn_unsorted,
        "pred_dur": pred_dur_unsorted,
        "order": order,  # sorted order used internally
    }
