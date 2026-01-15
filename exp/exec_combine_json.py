from exp_utils import combine_jsons, replace_certain_key_value



if __name__ == '__main__':
    # combine
    attn_json1 = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v6/psdcond_mdit_random_v1.json"
    attn_json2 = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v6/psdcond_mdit_random_v2.json"
    combined_attn_json = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v6/psdcond_mdit_random.json"
    #combine_jsons(attn_json1, attn_json2, combined_attn_json)

    # psdave replace
    original_dict_path = "/home/rosen/ckpt/exp/mdit_tts_esd/psdave_monoDiT_DiT_drawspeech_styletts2_hierspeech_reference.json"
    replacement_dict_path = "/home/rosen/ckpt/exp/mdit_tts_esd/psdave_monoDiT_reference.json"
    replaced_dict_path = "/home/rosen/ckpt/exp/mdit_tts_esd/psdave_monoDiT_DiT_drawspeech_styletts2_hierspeech_reference_v2.json"  # replace with ablation best

    # psd replace
    # v5=ab0808_m01, v4=ab0808 v3=0007 v2=0307
    original_dict_path = "/home/rosen/ckpt/exp/mdit_tts_esd/psd_monoDiT_DiT_drawspeech_styletts2_hierspeech_reference.json"
    replacement_dict_path = "/home/rosen/ckpt/exp/mdit_tts_esd_multiversion/psd_monoDiT_reference.json"
    replaced_dict_path = "/home/rosen/ckpt/exp/mdit_tts_esd/psd_monoDiT_DiT_drawspeech_styletts2_hierspeech_reference_v6_ab0000.json"  # replace with ablation best

    replace_certain_key_value(original_dict_path, replacement_dict_path, replaced_dict_path=replaced_dict_path, key_depth=2, key_name="monoDiT")
    # replace