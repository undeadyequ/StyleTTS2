import os.path
import torch
import torchaudio
from exec_draw_two_pitch import extract_f0, extract_energy_spectral
from inferenceAPI_bertFusion import InferenceAPI

model_root_dir = "/home/rosen/ckpt/styletts2_libriTTS/"
model_config = {
    "styletts2_txt2mel": ["first_txt2mel/epoch_1st_00048.pth", "first_txt2mel/config_libritts_txt2mel.yml",
                          "first_txt2mel/epoch_2nd_00028.pth", "first_txt2mel/config_libritts_txt2mel.yml"],
    "mdit_cfm_v10": ["", "",
                     "first_txt2mel_cfm_v10/epoch_2nd_00048.pth",
                     "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"],
    "decodit_cfm_v29": ["", "",
                        "first_txt2mel_cfm_v29/epoch_2nd_00048.pth",
                        "first_txt2mel_cfm_v29/config_libritts_txt2mel_cfm_v29.yml"]}

def synthesize_semStyle_fuseSytle_sample(model_name = "decodit_cfm_v29"):
    api = InferenceAPI(
        model_name=model_name,
        ckpt_path=model_root_dir + model_config[model_name][2],
        config_path=model_root_dir + model_config[model_name][3],
        device="cuda")
    # pitch cmp: spk0019_Surprise_ref3
    # energy cmp: spk0019_Angry_ref4
    reference_wav3 = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0019_Surprise_ref1.wav"
    reference_wav3_lab = "/home/rosen/ckpt/exp2/mdit_tts_esd/reference/random/spk0019_Surprise_ref1.lab"
    text = "I must have two to fetch and carry."
    # out
    out_dir = "/home/rosen/ckpt/exp/mdit_tts_esd/img_out/psdcond_syn_gamma"
    cond_syn_psd = {}

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # 1. Single Inference
    # read ref txt
    with open(reference_wav3_lab) as f:
        reference_wav3_lab = f.read()

    # Base model: trend
    torch.manual_seed(0)
    base_name = "fuseStyle"
    out_wav_trend = f"{out_dir}/{model_name}_{base_name}.wav"  # _add10  _05strength
    wav_trd, F0_pred_trend, F0_ref, N_pred_trend, N_real, trd_index = api.synthesize_one(
        text, reference_wav3, reference_wav3_lab, drop_trend=False, return_pitch=True, cfg_strength=3.0, return_trd_index=True)
    torchaudio.save(out_wav_trend, torch.from_numpy(wav_trd).unsqueeze(0), sample_rate=24000)

    # noTrend -> add quantized data
    torch.manual_seed(0)
    variant2 = "semStyle"
    out_wav_variant2 = f"{out_dir}/{model_name}_{variant2}.wav"  # notrend
    wav_notrd, F0_pred_variant2, _, N_pred_variant2, _ = api.synthesize_one(
        text, reference_wav3, reference_wav3_lab, drop_trend=True, return_pitch=True, cfg_strength=3.0)
    torchaudio.save(out_wav_variant2, torch.from_numpy(wav_notrd).unsqueeze(0), sample_rate=24000)
    # uv_masks_frame = F.interpolate(uv_masks_frame.unsqueeze(0), scale_factor=2, mode="nearest").squeeze(0)

    _, F0_syn_trend = extract_f0(out_wav_trend)
    _, N_syn_trend = extract_energy_spectral(out_wav_trend)

    _, F0_syn_variant2 = extract_f0(out_wav_variant2)
    _, N_syn_variant2 = extract_energy_spectral(out_wav_variant2)

    # Helper to convert tensor/numpy to list for JSON serialization
    def to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().squeeze().numpy().tolist()
        elif hasattr(x, 'tolist'):  # numpy array
            return x.tolist()
        return x

    F0_syn_trend = to_list(F0_syn_trend)
    N_syn_trend = to_list(N_syn_trend)
    F0_syn_variant2 = to_list(F0_syn_variant2)
    N_syn_variant2 = to_list(N_syn_variant2)

    # Save result - convert all values to lists for JSON serialization
    cond_syn_psd["cond"] = {
        "fusStyle": [to_list(F0_pred_trend), to_list(N_pred_trend)],
        "posStyle": [to_list(trd_index)],
        "semStyle": [to_list(F0_pred_variant2), to_list(N_pred_variant2)],
        "refStyle": [to_list(F0_ref), to_list(N_real)],
    }
    cond_syn_psd["syn"] = {
        "fusStyle": [F0_syn_trend, N_syn_trend],
        "semStyle": [F0_syn_variant2, N_syn_variant2],
    }
    return cond_syn_psd