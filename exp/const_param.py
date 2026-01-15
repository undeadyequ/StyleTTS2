# Experiment data
logs_dir_par = "/home/rosen/Project/Speech-Backbones/GradTTS/logs/"
#logs_dir = logs_dir_par + "gradtts_crossSelf_v2/"
#logs_dir = logs_dir_par + "interpEmoTTS_frame2frameAttn_noJoint/"
logs_dir = logs_dir_par + "styleEnhancedTTS_stditMocha_norm_hardMonMask/" #"interpEmoTTS_frame2binAttn_noJoint/"

config_dir = "/home/rosen/Project/Speech-Backbones/GradTTS/config/ESD"

# Speech dataset for training
esd_processed_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD"
melstyle_dir = esd_processed_dir + "/emo_reps"
psd_quants_dir = esd_processed_dir + "/psd_quants"
psd_dir = "/home/rosen/Project/FastSpeech2/preprocessed_data/ESD/pitch"
wav_dir = "/home/rosen/Project/FastSpeech2/ESD/16k_wav"
textgrid_dir = esd_processed_dir + "/TextGrid"


## SER dataset
label2id_SER = {
    "angry": 0,
    "calm": 3,
    "disgust": 0,
    "fearful": 5,
    "happy": 4,
    "neutral": 3,
    "sad": 2,
    "surprised": 1
}

emo_num_dict = {
    "Angry": 0,
    "Surprise": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4
}


# Reference data
##
emo_melstyle_dict = {
    "Angry": "0015_000415.npy",  # Tom could hardly speak for laughing
    "Surprise": "0015_001465.npy",
    "Sad": "0015_001115.npy",  # Tom could hardly speak for laughing
    "Neutral": "0015_000065.npy",
    "Happy": "0015_000765.npy"
}

# Said the American to Chinese.
emo_melstyle_dict1 = {
    "Angry": "0019_000401.npy",
    "Surprise": "0019_001451.npy",
    "Sad": "0019_001101.npy",
    "Neutral": "0019_000051.npy",
    "Happy": "0019_000751.npy"
}

# He was still(high_energy) in the forest! -> H EI S T il
emo_melstyle_dict2 = {
    "Angry": "0019_000403.npy",
    "Surprise": "0019_001453.npy",
    "Sad": "0019_001103.npy",
    "Neutral": "0019_000053.npy",
    "Happy": "0019_000753.npy"
}

emo_melstyle_list_dict2 = {
    "Angry": ["0019_000403.npy", ""],
    "Surprise": "0019_001453.npy",
    "Sad": "0019_001103.npy",
    "Neutral": "0019_000053.npy",
    "Happy": "0019_000753.npy"
}

emo_melstyleSpk_dict = {
    "Angry": ("0019_000403.npy", 19),
    "Surprise": ("0019_001453.npy", 19),
    "Sad": ("0019_001103.npy", 19),
    "Neutral": ("0019_000053.npy", 19),
    "Happy": ("0019_000753.npy", 19)
}

# used for extract pitch (Phoneme average)
psd_dict = {
    "Angry": "0015-pitch-0015_000415.npy",  # Tom could hardly speak for laughing
    "Surprise": "0015-pitch-0015_001465.npy",
    "Sad": "0015-pitch-0015_001115.npy",  # Tom could hardly speak for laughing
    "Neutral": "0015-pitch-0015_000065.npy",
    "Happy": "0015-pitch-0015_000765.npy"
}

# used for extract pitch (No phoneme average)
wav_dict = {
    "Angry": "0015_000415.wav",  # Tom could hardly speak for laughing
    "Surprise": "0015_001465.wav",
    "Sad": "0015_001115.wav",  # Tom could hardly speak for laughing
    "Neutral": "0015_000065.npy",
    "Happy": "0015_000765.npy"
}

model_meta = {
        "styletts2": ["/home/rosen/ckpt/styletts2_libriTTS", "epochs_2nd_00020.pth", "config.yml"],
        "styletts2_txt2mel": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel/epoch_2nd_00028.pth", "first_txt2mel/config_libritts_txt2mel.yml"],
        "drawspeech": ["root_dir", "", ""],
        "hierspeech": ["", ""],
        "DiT": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_dit_v1/epoch_2nd_00048.pth", "first_txt2mel_cfm_dit_v1/config_libritts_txt2mel_cfm_dit_v1.yml"],
        # monoDiT multi version
        "monoDiT": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_v10/epoch_2nd_00048.pth", "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "monoDiT_ab0307": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_v10/epoch_2nd_00048.pth", "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "monoDiT_ab0007": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_v10/epoch_2nd_00048.pth",
                       "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "monoDiT_ab0000": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_v10/epoch_2nd_00048.pth",
                           "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "decoDiT_v16": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_v16/epoch_2nd_00020.pth", "first_txt2mel_cfm_v16/config_libritts_txt2mel_cfm_v16.yml"],
        "monoDiT_ab0808_m08_fb03": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_v10/epoch_2nd_00048.pth", "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "monoDiT_ab0307_m08_fb03": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_v10/epoch_2nd_00048.pth", "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
        "monoDiT_ab0808_m08_fbnone": ["/home/rosen/ckpt/styletts2_libriTTS", "first_txt2mel_cfm_v10/epoch_2nd_00048.pth", "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"]
}

model_infer_config = {
        "styletts2": {
            "alpha": 0.8,
            "beta": 0.8,
            "diffusion_steps": 10,
            "embedding_scale": 1,
        },
        "DiT": {
            "alpha": 1,
            "beta": 1,
            "diffusion_steps": 10,
            "embedding_scale": 1,
            "style_dim": 256,
            "mix_ref_pe_type": "none",  # hyper
            "cfg_strength": 3,
            "mono_guide_delta": -1.0,  # monoDiT
            "Vis_F0": False,
            "fuse_beta": 0.3     # reference aware
        },
        "styletts2_txt2mel": {},
        "drawspeech": {},
        "hierspeech": {},
        "monoDiT": {
            "alpha": 0.3,
            "beta": 0.1,
            "diffusion_steps": 10,
            "embedding_scale": 1,
            "style_dim": 256,
            "mix_ref_pe_type": "none",  # ref_pred_add none ref_pred_gate
            "cfg_strength": 3,
            "mono_guide_delta": -1.0,
            "Vis_F0": True,
            "fuse_beta": 0.3
        },
        "monoDiT_ab0307": {
            "alpha": 0.3,
            "beta": 0.7,
            "diffusion_steps": 10,
            "embedding_scale": 1,
            "style_dim": 256,
            "mix_ref_pe_type": "none",  # ref_pred_add none ref_pred_gate
            "cfg_strength": 3,
            "mono_guide_delta": 0.8,
            "Vis_F0": False,
            "fuse_beta": 0.3
        },
        "monoDiT_ab0007": {
            "alpha": 0,
            "beta": 0.7,
            "diffusion_steps": 10,
            "embedding_scale": 1,
            "style_dim": 256,
            "mix_ref_pe_type": "none",  # ref_pred_add none ref_pred_gate
            "cfg_strength": 3,
            "mono_guide_delta": 0.8,
            "Vis_F0": False,
            "fuse_beta": 0.3
        },
        "monoDiT_ab0000": {
            "alpha": 0,
            "beta": 0,
            "diffusion_steps": 10,
            "embedding_scale": 1,
            "style_dim": 256,
            "mix_ref_pe_type": "none",  # ref_pred_add none ref_pred_gate
            "cfg_strength": 3,
            "mono_guide_delta": -0.1,
            "Vis_F0": False,
            "fuse_beta": 0.3
        },
        "decoDiT_v16": {
            "alpha": 0.3,
            "beta": 0.7,
            "diffusion_steps": 10,
            "embedding_scale": 1,
            "cfg_strength": 3,
            "Vis_F0": False,
        },
        "monoDiT_ab0808_m08_fb03": {
            "alpha": 0.8,
            "beta": 0.8,
            "diffusion_steps": 10,
            "embedding_scale": 1,
            "style_dim": 256,
            "mix_ref_pe_type": "ref_pred_add",  # ref_pred_add none ref_pred_gate
            "cfg_strength": 3,
            "mono_guide_delta": 0.8,
            "Vis_F0": False,
            "fuse_beta": 0.3
        },

        "monoDiT_ab0808_m08_fbnone": {
            "alpha": 0.8,
            "beta": 0.8,
            "diffusion_steps": 10,
            "embedding_scale": 1,
            "style_dim": 256,
            "mix_ref_pe_type": "none",  # ref_pred_add none ref_pred_gate
            "cfg_strength": 3,
            "mono_guide_delta": 0.8,
            "Vis_F0": False,
            "fuse_beta": 0.3
        },
}