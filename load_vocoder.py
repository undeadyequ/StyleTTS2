
from attrdict import AttrDict
from Modules.hifi_gan.vocoder import Generator
import json

import warnings

warnings.simplefilter('ignore')
import torch
import os
import glob

def load_checkpoint_vocoder(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def get_vocoder(ckpt_dir="/home/rosen/ckpt/styletts/Vocoder/LibriTTS/", device=None):
    # load vocoder
    cp_g = scan_checkpoint(ckpt_dir, 'g_')
    config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint_vocoder(cp_g, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

