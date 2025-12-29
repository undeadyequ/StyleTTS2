import json
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessor.preprocessor_v2 import PreprocessorExtract

import time
from dataclasses import asdict
import argparse
from exp.mel_config import MelConfig          ################ BE CAREFUL; Must same with draw?.yaml #########
from utilities.audio.audio_for_eval import LogMelSpectrogram, load_and_resample_audio, PitEngExtractor, load_audio

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

def extract_psd(mel_config, out_speech_dir, model_n="unkown", save_psd_file="", prosody_dict=dict()):
    pitch_extractor = PitEngExtractor(**asdict(mel_config()), need_energy=True)
    speech_list = [speech for speech in os.listdir(out_speech_dir) if speech.endswith(".wav")]
    if len(prosody_dict.keys()) == 0:
        prosody_dict = {}
    for speech in speech_list:
        speech_f = os.path.join(out_speech_dir, speech)
        spk, emo_id = speech.split("_")[:2]
        wav = load_audio(speech_f, device=device)
        if spk not in prosody_dict.keys():
            prosody_dict[spk] = dict()
        if emo_id not in prosody_dict[spk].keys():
            prosody_dict[spk][emo_id] = dict()
        if model_n not in prosody_dict[spk][emo_id].keys():
            prosody_dict[spk][emo_id][model_n] = {
                "pitch": [],
                "energy": [],
                "speechid": []}
        try:
            pitch, energy = pitch_extractor.forward(wav)  # [2, time // hop_length]
        except IOError:
            pitch, energy = torch.zeros(1), torch.zeros(1)
            print("{} is failed to extract psd".format(speech_f))
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
    pitch_extractor = PitEngExtractor(**asdict(mel_config()), need_energy=True)
    speech_list = [speech for speech in os.listdir(out_speech_dir) if speech.endswith(".wav")]
    if len(prosody_dict.keys()) == 0:
        prosody_dict = {}
    for speech in speech_list:
        speech_f = os.path.join(out_speech_dir, speech)
        spk, emo_id = speech.split("_")[:2]
        wav = load_audio(speech_f, device=device)
        prosody_dict.setdefault(spk, {})
        prosody_dict[spk].setdefault(emo_id, {})
        prosody_dict[spk][emo_id].setdefault(model_n, {})
        prosody_dict[spk][emo_id][model_n].setdefault(fine_cate, {"pitch": [], "energy": [], "speechid": []})
        try:
            pitch, energy = pitch_extractor.forward(wav)  # [2, time // hop_length]
        except IOError:
            pitch, energy = torch.zeros(1), torch.zeros(1)
            print("{} is failed to extract psd".format(speech_f))
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
