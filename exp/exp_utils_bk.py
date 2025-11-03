import torch
from const_param import emo_num_dict
import os
import numpy as np

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

def get_emo_label(emo: str,
                  emo_num_dict: dict=emo_num_dict,
                  emo_type="index",
                  emo_value=1,
                  emo_num=5
                  ):
    emo_tensor = torch.tensor([emo_num_dict[emo]], dtype=torch.long).cuda()
    if emo_type == "onehot":
        emo_tensor = torch.nn.functional.one_hot(emo_tensor, num_classes=emo_num).cuda()
    """
    emo_emb_dim = len(emo_num_dict.keys())
    emo_label = [[0] * emo_emb_dim]
    emo_label[0][emo_num_dict[emo]] = emo_value
    """
    return emo_tensor

def get_synStyle_from_file(synStyle_f,
                           split_char="|",
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
            spk = speech_path.split("/")[4]
            emo = "Neutral"
            styles.append((spk, emo, txt, speech_path))
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
