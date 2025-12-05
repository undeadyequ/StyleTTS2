import argparse
from inference_1_or_2 import syn_speech_by_second_model, get_second_model, get_synStyle_from_file, get_synText_from_file
from inference_1_or_2 import model_config, model_root_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_prosody_fuse",
        type=bool,
        required=False,
        default=False,  # drawspeech_ljspeech_dit_22k_v1
        help="reference-aware prosody fuse experiment",
    )
    parser.add_argument(
        "--exp_bandwidth_sigma",
        type=bool,
        required=False,
        default=False,  # drawspeech_ljspeech_dit_22k_v1
        help="reference-aware prosody fuse experiment",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        default="esd",  # drawspeech_ljspeech_dit_22k_v1
        help="test on esd",
    )
    parser.add_argument(
        "--styles_f",
        type=str,
        required=False,
        default="exp/data/libri_r1.txt",  # drawspeech_ljspeech_dit_22k_v1
    )
    parser.add_argument(
        "--txt_f",
        type=str,
        required=False,
        default="exp/data/libri_r1.txt",  # drawspeech_ljspeech_dit_22k_v1
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="mdit_cfm_v10",  # drawspeech_ljspeech_dit_22k_v1
    )
    args = parser.parse_args()
    ## Condition
    alpha, beta = 1, 1
    model_name = "mdit_cfm_v10"  # "styletts2_txt2mel"  "mdit_cfm"
    second_model_path, second_config = model_root_dir + model_config[model_name][2], model_root_dir + model_config[model_name][3]
    second_model, sampler, model_params = get_second_model(ckpt=second_model_path, config_f=second_config, model_name=model_name)

    ## INPUT
    syn_styles = get_synStyle_from_file(args.style_f, split_char='|', dataset_name=args.dataset_name)  # emotion changed
    synTexts = get_synText_from_file(args.txt_f)

    if args.exp_prosody_fuse:
        mix_ref_pe_types = ["none", "ref_pred_gate"]
        for mix_ref_pe_type in mix_ref_pe_types:
            syn_speech_by_second_model(synTexts, syn_styles, out_dir, second_model=second_model, sampler=sampler,
                                       model_params=model_params, alpha=alpha, beta=beta, style_dim=style_dim,
                                       mix_ref_pe_type=mix_ref_pe_type, reference_dir=f"reference_{dataset}",
                                       cfg_strength=cfg_strength)
    elif args.exp_bandwidth_sigma:
        bd_sigmas = [0.2, 0.5, 0.8]
        for bd_sigma in bd_sigmas:
            syn_speech_by_second_model(synTexts, syn_styles, out_dir, second_model=second_model, sampler=sampler,
                                       model_params=model_params, alpha=alpha, beta=beta, style_dim=style_dim,
                                       mix_ref_pe_type=mix_ref_pe_type, reference_dir=f"reference_{dataset}",
                                       cfg_strength=cfg_strength, bd_sigma=bd_sigma)
    else:
        syn_speech_by_second_model(synTexts, syn_styles, out_dir, second_model=second_model, sampler=sampler,
                                   model_params=model_params, alpha=alpha, beta=beta, style_dim=style_dim,
                                   mix_ref_pe_type=mix_ref_pe_type, reference_dir=f"reference_{dataset}",
                                   cfg_strength=cfg_strength, bd_sigma=bd_sigma)


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