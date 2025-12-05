import shutil
import os
import sys
from tqdm import tqdm
import argparse
import yaml
import torch
import json
os.chdir("/home/rosen/Project/DrawSpeech_PyTorch")
sys.path.append("/home/rosen/Project/DrawSpeech_PyTorch")

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from utilities.tools import get_restore_step
from utilities.model_util import instantiate_from_config
from utilities.tools import build_dataset_json_from_list
from drawspeech.conditional_models import *
from drawspeech.utilities.data.dataset import AudioDataset

def set_cond_infer_mode(latent_diffusion):
    for key in latent_diffusion.cond_stage_model_metadata.keys():
        model_idx = latent_diffusion.cond_stage_model_metadata[key]["model_idx"]
        if isinstance(latent_diffusion.cond_stage_models[model_idx], TextEncoderwithVarianceAdaptor):
            print("Set infer mode for TextEncoderwithVarianceAdaptor")
            latent_diffusion.cond_stage_models[model_idx].infer = True
        if isinstance(latent_diffusion.cond_stage_models[model_idx], SketchEncoder):
            print("Set infer mode for SketchEncoder")
            latent_diffusion.cond_stage_models[model_idx].infer = True

    return latent_diffusion

def infer_exp(dataset_json, configs, config_yaml_path, exp_group_name, exp_name, syn_styles, synTexts,
              out_speech_dir=None, ref_dir=None, out_attn_dir=None, batch_size=1, save_attn=False):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)
    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])
    log_path = configs["log_directory"]
    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []
    val_dataset = AudioDataset(
        configs, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_json
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
    )

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None
    # get checkpoint from yaml path
    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")
    #wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)
    out_ori_dir = "/".join(out_speech_dir.split("/")[:-1])
    shutil.copy(config_yaml_path, out_ori_dir)

    if config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    elif len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    guidance_scale = configs["model"]["params"]["evaluation_params"][
        "unconditional_guidance_scale"
    ]
    ddim_sampling_steps = configs["model"]["params"]["evaluation_params"][
        "ddim_sampling_steps"
    ]
    n_candidates_per_samples = configs["model"]["params"]["evaluation_params"][
        "n_candidates_per_samples"
    ]

    checkpoint = torch.load(resume_from_checkpoint)
    latent_diffusion.load_state_dict(checkpoint["state_dict"])

    #latent_diffusion = set_cond_infer_mode(latent_diffusion)
    
    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()

    # copy reference
    copy_reference_speech(dataset_json, ref_dir)

    # read each sample info of batch into attn_dict (expect q_dur)
    # attn_dict, wav_names = read_attn_info_from_batch(val_loader)
    # _, q_dur = generate_with_extr_out(wav_names)

    # synthesize speech
    if "dit" in os.path.basename(config_yaml_path):
        wav_out, attn_dict = latent_diffusion.generate_sample_with_extra_out(
            val_loader,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_sampling_steps,
            n_gen=n_candidates_per_samples,
            name=out_speech_dir,
            save_attn=save_attn
        )
        return wav_out, attn_dict
    else:
        waveform_save_path, attn_dict = latent_diffusion.generate_sample_with_extra_out(
            val_loader,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_sampling_steps,
            n_gen=n_candidates_per_samples,
            name=out_speech_dir
        )
        return waveform_save_path, attn_dict

def read_attn_info_from_batch(batchs):
    for i, batch in enumerate(batchs):
        pass


def copy_reference_speech(infer_json, ref_speech_dir):
    for d in tqdm(infer_json["data"], desc="Copy reference"):
        basename = d["wav"].split("/")[-1].replace(".wav", "")
        speech_path = d["wav"]
        spk, emo, r_id, ref_txt = d["spk"], d["emo"], d["r_id"], d["ref_transcription"]
        ref_speech_id = f'spk{spk}_{emo}_ref{r_id}'
        src_ref_txt_f = f'{ref_speech_dir}/{ref_speech_id}.lab'

        with open(src_ref_txt_f, "w") as file1:
            file1.write(ref_txt)
        dst_wav = f'{ref_speech_dir}/{ref_speech_id}.wav'   # renamed
        shutil.copyfile(speech_path, dst_wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        default="drawspeech/config/drawspeech_ljspeechesd_dit_16k.yaml",  # drawspeech_ljspeech_dit_22k_v1
        help="path to config .yaml file",
    )
    parser.add_argument(
        "-l",
        "--list_inference",
        type=str,
        required=False,
        default="esd_test_unparallel.json",   # ljspeech_test
        help="The filelist that contain captions (and optionally filenames)",
    )
    parser.add_argument(
        "-reload_from_ckpt",
        "--reload_from_ckpt",
        type=str,
        required=False,
        help="the checkpoint path for the model",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml
    if args.list_inference.endswith(".json"):
        dataset_json = json.load(open(args.list_inference, "r"))
    else:
        dataset_json = build_dataset_json_from_list(args.list_inference)
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt != None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    if "pitch" in dataset_json.keys() and dataset_json["pitch"] != "":
        config_yaml["preprocessing"]["preprocessed_data"]["pitch"] = dataset_json["pitch"]
    if "energy" in dataset_json.keys() and dataset_json["energy"] != "":
        config_yaml["preprocessing"]["preprocessed_data"]["energy"] = dataset_json["energy"]
    if "duration" in dataset_json.keys() and dataset_json["duration"] != "":
        config_yaml["preprocessing"]["preprocessed_data"]["duration"] = dataset_json["duration"]

    infer_exp(dataset_json, config_yaml, config_yaml_path, exp_group_name, exp_name)