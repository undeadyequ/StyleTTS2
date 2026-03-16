import json
import torch
import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessor.preprocessor_v2 import PreprocessorExtract

import time
from dataclasses import asdict
import argparse
from exp.mel_config import MelConfig          ################ BE CAREFUL; Must same with draw?.yaml #########
from utilities.audio.audio_for_eval import LogMelSpectrogram, load_and_resample_audio, PitEngExtractor, load_audio
from utils import length_to_mask, mask_from_lens, maximum_path
import torchaudio
import librosa

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def mta_speech_bk(out_sub_dir, tg_dir):
    ## Conduct alignment by "python mfa_dir.py ~/data/test_data/ english_mfa ~/data/tg/" under aligner conda enviroment
    command = "/home/rosen/anaconda3/envs/aligner/bin/python mfa_dir.py {} english_mfa {}".format(out_sub_dir, tg_dir)
    print(command)
    os.system(command)

def mta_speech(out_sub_dir, tg_dir):
    ## Conduct alignment by "python mfa_dir.py ~/data/test_data/ english_mfa ~/data/tg/" under aligner conda enviroment
    print("####################")
    print("Please conduct alignment by: 1> conda activate aligner\n")
    print("2> cd Project/StableTTS/exp/")
    print("3> python mfa_dir.py {} english_mfa {}".format(out_sub_dir, tg_dir))
    print("####################")

def extract_psd_single(wav_path, mel_config=None, pitch_extractor=None, device=None):
    """
    Extract pitch and energy from a single wav file.

    Args:
        wav_path: Path to the wav file.
        mel_config: Mel configuration class or instance. Required if pitch_extractor is None.
        pitch_extractor: Pre-initialized PitEngExtractor. If None, creates one from mel_config.
        device: Torch device. If None, uses CUDA if available.

    Returns:
        pitch: Tensor of pitch values.
        energy: Tensor of energy values.
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if pitch_extractor is None:
        if mel_config is None:
            mel_config = MelConfig
        # Handle both class and instance
        config_dict = asdict(mel_config()) if callable(mel_config) else asdict(mel_config)
        pitch_extractor = PitEngExtractor(**config_dict, need_energy=True)

    wav = load_audio(wav_path, device=device)
    try:
        pitch, energy = pitch_extractor.forward(wav)
    except TypeError:
        pitch, energy = torch.zeros(1), torch.zeros(1)
        print("{} failed to extract psd".format(wav_path))

    return pitch, energy


def extract_psd(mel_config, out_speech_dir, model_n="unkown", save_psd_file="", prosody_dict=dict()):
    # Handle both class and instance
    config_dict = asdict(mel_config()) if callable(mel_config) else asdict(mel_config)
    pitch_extractor = PitEngExtractor(**config_dict, need_energy=True)

    speech_list = [speech for speech in os.listdir(out_speech_dir) if speech.endswith(".wav")]
    if len(prosody_dict.keys()) == 0:
        prosody_dict = {}
    for speech in speech_list:
        speech_f = os.path.join(out_speech_dir, speech)
        spk, emo_id = speech.split("_")[:2]
        if spk not in prosody_dict.keys():
            prosody_dict[spk] = dict()
        if emo_id not in prosody_dict[spk].keys():
            prosody_dict[spk][emo_id] = dict()
        if model_n not in prosody_dict[spk][emo_id].keys():
            prosody_dict[spk][emo_id][model_n] = {
                "pitch": [],
                "energy": [],
                "speechid": []}
        pitch, energy = extract_psd_single(speech_f, pitch_extractor=pitch_extractor, device=device)
        prosody_dict[spk][emo_id][model_n]["pitch"].append(pitch.tolist())
        prosody_dict[spk][emo_id][model_n]["energy"].append(energy.tolist())
        prosody_dict[spk][emo_id][model_n]["speechid"].append(speech.split(".")[0])
    # save psd json
    if len(save_psd_file) != 0:
        with open(save_psd_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
    return prosody_dict


def extract_psdave(mel_config, cmp_modelnames, out_dir=None):
    """
    # Extract PSD given wav/tgt and save json
    Args:
        configs1:
    Returns:
    """
    prosody_dict = dict()
    preprocessor = PreprocessorExtract(**asdict(mel_config()))

    # Do PSD extraction
    for i, model_n in enumerate(cmp_modelnames):
        out_speech_dir = os.path.join(out_dir, model_n, "random")  # audio_txt folder
        out_mfa_dir = os.path.join(out_dir, model_n, "mfa")  # textgrid folder
        out_psd_dir = os.path.join(out_dir, model_n, "psd")  # psd folder
        # DO MFA
        if not os.path.isdir(out_mfa_dir):
            mta_speech(out_speech_dir, out_mfa_dir)
            if i == len(cmp_modelnames):
                sys.exit()
            continue

        speech_list = [speech for speech in os.listdir(out_speech_dir) if speech.endswith(".wav")]
        speech_list.sort()
        print("Do psd extraction!")
        for speech in speech_list:                            # spk0013_Angry_ref0_syn1.wav
            speech_f = os.path.join(out_speech_dir, speech)
            spk, emo_id = speech.split("_")[:2]
            ######## CHECK1
            #if speech != "spk19_Surprise_txt0.wav":   # high pitch at last word!
            #    continue
            if spk not in prosody_dict.keys():
                prosody_dict[spk] = dict()
            if emo_id not in prosody_dict[spk].keys():
                prosody_dict[spk][emo_id] = dict()
            if model_n not in prosody_dict[spk][emo_id].keys():
                prosody_dict[spk][emo_id][model_n] = {
                    "phonemes": [],
                    "pitch": [],
                    "energy": [],
                    "duration": [],
                    "speechid": []}
            tg_path = os.path.join(out_mfa_dir, "{}.TextGrid".format(os.path.basename(speech_f).split(".")[0]))
            try:
                phonemes, pitch, energy, mels, duration = preprocessor.extract_pitch_energy_mel(speech_f, tg_path=tg_path, out_dir=out_psd_dir, average_phoneme=True, save_npy=False)
            except IOError:
                print("{} is failed to extract psd".format(speech_f))
            prosody_dict[spk][emo_id][model_n]["phonemes"].append(phonemes.split(" "))
            prosody_dict[spk][emo_id][model_n]["pitch"].append(pitch.tolist())
            prosody_dict[spk][emo_id][model_n]["energy"].append(energy.tolist())
            prosody_dict[spk][emo_id][model_n]["duration"].append(duration)
            prosody_dict[spk][emo_id][model_n]["speechid"].append(speech.split(".")[0])
    return prosody_dict

def extract_psd_fine_class2(mel_config, out_speech_dir, model_n="unkown", save_psd_file="", prosody_dict=dict(), fine_cate="fine"):
    # Handle both class and instance
    config_dict = asdict(mel_config()) if callable(mel_config) else asdict(mel_config)
    pitch_extractor = PitEngExtractor(**config_dict, need_energy=True)

    speech_list = [speech for speech in os.listdir(out_speech_dir) if speech.endswith(".wav")]
    if len(prosody_dict.keys()) == 0:
        prosody_dict = {}
    for speech in speech_list:
        speech_f = os.path.join(out_speech_dir, speech)
        spk, emo_id = speech.split("_")[:2]
        prosody_dict.setdefault(spk, {})
        prosody_dict[spk].setdefault(emo_id, {})
        prosody_dict[spk][emo_id].setdefault(model_n, {})
        prosody_dict[spk][emo_id][model_n].setdefault(fine_cate, {"pitch": [], "energy": [], "speechid": []})

        pitch, energy = extract_psd_single(speech_f, pitch_extractor=pitch_extractor, device=device)
        prosody_dict[spk][emo_id][model_n][fine_cate]["pitch"].append(pitch.tolist())
        prosody_dict[spk][emo_id][model_n][fine_cate]["energy"].append(energy.tolist())
        prosody_dict[spk][emo_id][model_n][fine_cate]["speechid"].append(speech.split(".")[0])
    # save psd json
    if len(save_psd_file) != 0:
        with open(save_psd_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(prosody_dict, sort_keys=True, indent=4))
    return prosody_dict


def extract_psd_fine_class(mel_config, model_name1, model_name2=None, out_dir=None, fine_categ_labels=("")):
    """
    # Extract PSD given wav/tgt and save json
    Args:
        configs1:
    Returns:
    """
    models_n = ["reference", model_name1, model_name2] if model_name2 is not None else ["reference", model_name1]

    prosody_dict = dict()
    preprocessor = PreprocessorExtract(**asdict(mel_config))
    # Do MFA
    for i, model_n in enumerate(models_n):
        ## Create output dir
        out_speech_dir = os.path.join(out_dir, model_n)  # audio_txt folder
        out_mfa_dir = os.path.join(out_dir, model_n + "_mfa")  # textgrid folder
        if not os.path.isdir(out_mfa_dir):
            mta_speech(out_speech_dir, out_mfa_dir)
        else:
            print("Escape mfa!")

    # Do PSD extraction
    for i, model_n in enumerate(models_n):
        out_speech_dir = os.path.join(out_dir, model_n)  # audio_txt folder
        out_mfa_dir = os.path.join(out_dir, model_n + "_mfa")  # textgrid folder
        if not os.path.isdir(out_mfa_dir):
            mta_speech(out_speech_dir, out_mfa_dir)

        for fine_label in fine_categ_labels:
            out_psd_dir = os.path.join(out_dir, model_n + f"_psd/{fine_label}")  # psd folder
            # root dir
            out_speech_fine_dir = os.path.join(out_speech_dir, fine_label) if model_n != "reference" else out_speech_dir
            out_mfa_fine_dir = os.path.join(out_mfa_dir, fine_label) if model_n != "reference" else out_mfa_dir

            speech_list = [speech for speech in os.listdir(out_speech_fine_dir) if speech.endswith(".wav")]
            print("Do psd extraction!")
            for speech in speech_list:                            # spk0013_Angry_ref0_syn1.wav
                speech_f = os.path.join(out_speech_fine_dir, speech)
                spk, emo_id = speech.split("_")[:2]
                ######## CHECK1
                #if speech != "spk19_Surprise_txt0.wav":   # high pitch at last word!
                #    continue
                if spk not in prosody_dict.keys():
                    prosody_dict[spk] = dict()
                if emo_id not in prosody_dict[spk].keys():
                    prosody_dict[spk][emo_id] = dict()
                if model_n not in prosody_dict[spk][emo_id].keys():
                    prosody_dict[spk][emo_id][model_n] = {}
                if fine_label not in prosody_dict[spk][emo_id][model_n].keys():
                    prosody_dict[spk][emo_id][model_n][fine_label] = {
                        "phonemes": [],
                        "pitch": [],
                        "energy": [],
                        "duration": [],
                        "speechid": []
                    }
                tg_path = os.path.join(out_mfa_fine_dir, "{}.TextGrid".format(os.path.basename(speech_f).split(".")[0]))
                try:
                    phonemes, pitch, energy, mels, duration = preprocessor.extract_pitch_energy_mel(speech_f, tg_path=tg_path, out_dir=out_psd_dir, average_phoneme=True, save_npy=True)
                    ############ TEMP ###############
                except IOError:
                    print("{} is failed to extract psd".format(speech_f))

                prosody_dict[spk][emo_id][model_n][fine_label]["phonemes"].append(phonemes.split(" "))
                prosody_dict[spk][emo_id][model_n][fine_label]["pitch"].append(pitch.tolist())
                prosody_dict[spk][emo_id][model_n][fine_label]["energy"].append(energy.tolist())
                prosody_dict[spk][emo_id][model_n][fine_label]["duration"].append(duration)
                prosody_dict[spk][emo_id][model_n][fine_label]["speechid"].append(speech.split(".")[0])
    return prosody_dict

def extract_psd_from_speech_tgt(speech_f, tgt_f, out_dir, average_phoneme=True, save_npy=False, mel_config=None):
    """
    Extract PSD from a single speech and target file.
    Args:
        speech_f: Path to the speech file.
        tgt_f: Path to the target file (TextGrid).
        out_dir: Output directory for extracted PSD.
        average_phoneme: Whether to average phonemes.
        save_npy: Whether to save numpy arrays.
    Returns:
        prosody_dict: Dictionary containing extracted prosody features.
    """
    preprocessor = PreprocessorExtract(**asdict(mel_config))
    
    phonemes, pitch, energy, mels, duration = preprocessor.extract_pitch_energy_mel(
        speech_f, tg_path=tgt_f, out_dir=out_dir, average_phoneme=average_phoneme, save_npy=save_npy)
    
    prosody_dict = {
        "phonemes": phonemes.split(" "),
        "pitch": pitch.tolist(),
        "energy": energy.tolist(),
        "duration": duration,
        "speechid": os.path.basename(speech_f).split(".")[0]
    }
    return prosody_dict

########################################################################
# Phoneme-level pitch/energy standard deviation extraction
########################################################################

@torch.no_grad()
def _get_s2s(texts, input_lengths, mels, mel_input_length, text_aligner, device="cuda"):
    """Get monotonic s2s attention (force alignment) from text_aligner.
    Copied from inferenceAPI_bertFusion.get_s2s to avoid heavy import chain."""
    mask = length_to_mask(mel_input_length // 2).to(device)
    text_mask = length_to_mask(input_lengths).to(texts.device)
    _, _, s2s_attn = text_aligner(mels, mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)
    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // 2)
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    return s2s_attn_mono


def extract_phoneme_std_single(
    wav_path, lab_path, text_aligner, text_cleaner, global_phonemizer,
    to_mel, pitch_extractor=None, mel_config=None,
    mel_mean=-4, mel_std=4, device=None, min_voiced_frames=2,
):
    """Extract per-phoneme pitch std and energy std for a single wav file.

    Returns:
        pitch_stds: list[float] — one std per non-padding phoneme (voiced only for pitch)
        energy_stds: list[float] — one std per non-padding phoneme
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 1. Extract frame-level pitch & energy via PitEngExtractor
    pitch, energy = extract_psd_single(wav_path, mel_config=mel_config,
                                       pitch_extractor=pitch_extractor, device=device)
    pitch_np = pitch.cpu().numpy() if isinstance(pitch, torch.Tensor) else np.asarray(pitch)
    energy_np = energy.cpu().numpy() if isinstance(energy, torch.Tensor) else np.asarray(energy)

    # 2. Compute mel for text_aligner
    wave, sr = librosa.load(wav_path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    wave_tensor = torch.from_numpy(audio).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mel_mean) / mel_std
    mel_tensor = mel_tensor.to(device)
    mel_len = mel_tensor.size(-1)
    mel_tensor = mel_tensor[:, :, :(mel_len - mel_len % 2)]

    # 3. Tokenize text from .lab file
    from nltk.tokenize import word_tokenize
    text = open(lab_path).read().strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    tokens = text_cleaner(' '.join(ps))
    tokens.insert(0, 0)
    tokens.append(0)
    tokens_tensor = torch.LongTensor(tokens).to(device).unsqueeze(0)

    # 4. Force alignment via text_aligner
    input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(device)
    mel_input_length = torch.LongTensor([mel_tensor.shape[-1]]).to(device)
    s2s_attn_mono = _get_s2s(tokens_tensor, input_lengths, mel_tensor,
                             mel_input_length, text_aligner, device)

    # 5. Get phoneme durations (upscale from downsampled domain)
    durations = (s2s_attn_mono.sum(axis=-1) * 2).squeeze(0).long().cpu().tolist()

    # 6. Align frame counts
    total_dur = sum(durations)
    n_frames = min(total_dur, len(pitch_np), len(energy_np))

    # 7. Segment and compute std per phoneme
    pitch_stds, energy_stds = [], []
    frame_ptr = 0
    for idx, d in enumerate(durations):
        if frame_ptr + d > n_frames:
            d = n_frames - frame_ptr
        if d <= 0:
            break
        # Skip padding tokens (token 0 = silence)
        if tokens[idx] == 0:
            frame_ptr += d
            continue

        pitch_seg = pitch_np[frame_ptr:frame_ptr + d]
        energy_seg = energy_np[frame_ptr:frame_ptr + d]

        # Pitch: voiced frames only
        voiced = pitch_seg[pitch_seg > 0]
        if len(voiced) >= min_voiced_frames:
            pitch_stds.append(float(np.std(voiced)))

        # Energy: all frames
        if len(energy_seg) >= 2:
            energy_stds.append(float(np.std(energy_seg)))

        frame_ptr += d

    return pitch_stds, energy_stds


def extract_phoneme_std(
    mel_config, out_speech_dir, model_n="unknown", save_std_file="",
    std_dict=None, text_aligner=None, text_cleaner=None,
    global_phonemizer=None, asr_path=None, asr_config=None, device=None,
):
    """Extract phoneme-level pitch/energy std for all wav files in a directory.

    Returns:
        std_dict[spk][emo][model_n] = {
            "pitch_std": [[ph_std_1, ...], ...],
            "energy_std": [[ph_std_1, ...], ...],
            "speechid": [...]
        }
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if std_dict is None:
        std_dict = {}

    # Build tools once
    config_dict = asdict(mel_config()) if callable(mel_config) else asdict(mel_config)
    pitch_extractor = PitEngExtractor(**config_dict, need_energy=True)
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

    if text_cleaner is None:
        from text_utils import TextCleaner
        text_cleaner = TextCleaner()
    if global_phonemizer is None:
        import phonemizer as phon
        global_phonemizer = phon.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True)
    if text_aligner is None:
        from models import load_ASR_models
        text_aligner = load_ASR_models(asr_path, asr_config)
    text_aligner = text_aligner.to(device).eval()

    speech_list = sorted([f for f in os.listdir(out_speech_dir) if f.endswith(".wav")])
    print(f"Extracting phoneme-level std from {len(speech_list)} files in {out_speech_dir} ...")

    for i, speech in enumerate(speech_list):
        wav_path = os.path.join(out_speech_dir, speech)
        lab_path = os.path.join(out_speech_dir, speech.replace(".wav", ".lab"))
        if not os.path.exists(lab_path):
            print(f"  ⚠ No .lab file for {speech}, skipping")
            continue

        spk, emo_id = speech.split("_")[:2]
        std_dict.setdefault(spk, {})
        std_dict[spk].setdefault(emo_id, {})
        std_dict[spk][emo_id].setdefault(model_n, {
            "pitch_std": [], "energy_std": [], "speechid": []})

        try:
            pitch_stds, energy_stds = extract_phoneme_std_single(
                wav_path, lab_path, text_aligner, text_cleaner, global_phonemizer,
                to_mel, pitch_extractor=pitch_extractor, mel_config=mel_config,
                device=device)
            std_dict[spk][emo_id][model_n]["pitch_std"].append(pitch_stds)
            std_dict[spk][emo_id][model_n]["energy_std"].append(energy_stds)
            std_dict[spk][emo_id][model_n]["speechid"].append(speech.split(".")[0])
        except Exception as e:
            print(f"  ⚠ Failed {speech}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(speech_list)}")

    if save_std_file:
        with open(save_std_file, "w", encoding="utf-8") as f:
            json.dump(std_dict, f, sort_keys=True, indent=4)
        print(f"  ✓ Saved to {save_std_file}")

    return std_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract PSD from speech and target")
    parser.add_argument("--speech", type=str, required=False, help="speech file", default="/hdd/StableTTS/exp/result50_unpara_618_2_cutpad/test/wavtgt/spk0017_Sad_ref5_syn6_dit.wav")
    parser.add_argument("--tgt", type=str, required=False, help="textgrid file", default="/hdd/StableTTS/exp/result50_unpara_618_2_cutpad/test/wavtgt/spk0017_Sad_ref5_syn6_dit.TextGrid")
    parser.add_argument("--out_dir", type=str, required=False, help="Output directory for extracted PSD", default="/hdd/StableTTS/exp/result50_unpara_618_2_cutpad/test/wavtgt_res/")

    args = parser.parse_args()

    # Load mel config
    mel_config = MelConfig()

    # Extract PSD
    prosody_dict = extract_psd_from_speech_tgt(args.speech, args.tgt, args.out_dir, average_phoneme=False, save_npy=False, mel_config=mel_config)
    #print(json.dumps(prosody_dict, indent=4, ensure_ascii=False))
    print(prosody_dict)
