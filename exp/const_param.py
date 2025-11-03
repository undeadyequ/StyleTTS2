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