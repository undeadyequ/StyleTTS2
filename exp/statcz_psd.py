from pymcd.mcd import Calculate_MCD
import numpy as np
import os
from exp_utils import cut_pad_a2b_len_left
from dtaidistance import dtw
from statistics import mean
import json

mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
def interpolate_nan(array_like):
    array_like = np.array(array_like)
    array = array_like.copy()
    nans = np.isnan(array)
    def get_x(a):
        return a.nonzero()[0]
    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])
    return array


def statcz_psd_fine_mcd(prosody_dict):
    """
    For parallel test
    Args:
        prosody_dict:  {"spk": {"emo": {"A/B/R": {"05": {"psd":...}}}}}   <- 02 06 10 14 18 22
    Returns:
        psd_mcd_stat_res: {"spk": {"emo": {"A/B": {"05": [p1, e1]}}}}    <- no need m1
    """
    psd_mcd_stat_res = {}
    mean_func = lambda x: np.mean(np.array(x))

    for spk, emo_model_psd in prosody_dict.items():
        if spk not in psd_mcd_stat_res.keys():
            psd_mcd_stat_res[spk] = dict()
        for emo, model_psd in emo_model_psd.items():
            if emo not in psd_mcd_stat_res[spk].keys():
                psd_mcd_stat_res[spk][emo] = dict()
            for model_n, fine_psd_phone_sid in model_psd.items():
                if model_n != "reference":
                    if model_n not in psd_mcd_stat_res[spk][emo].keys():
                        psd_mcd_stat_res[spk][emo][model_n] = dict()
                    for fine_categ, psd_phone_sid in fine_psd_phone_sid.items():
                        if fine_categ not in psd_mcd_stat_res[spk][emo][model_n].keys():
                            psd_mcd_stat_res[spk][emo][model_n][fine_categ] = []
                        # statcz psd
                        wav_num = len(psd_phone_sid["pitch"])
                        p_diffs = []
                        e_diffs = []
                        for i in range(wav_num):
                            ref_i = int(psd_phone_sid["speechid"][i].split("_")[-2][-1])
                            syn_pitch_contour, ref_pitch_contour = interpolate_nan(psd_phone_sid["pitch"][i]), interpolate_nan(prosody_dict[spk][emo]["reference"][fine_categ]["pitch"][ref_i])
                            syn_energy_contour, ref_energy_contour = interpolate_nan(psd_phone_sid["energy"][i]), interpolate_nan(prosody_dict[spk][emo]["reference"][fine_categ]["energy"][ref_i])
                            p_diff, _ = dtw_sim_score(syn_pitch_contour, ref_pitch_contour)
                            e_diff, _ = dtw_sim_score(syn_energy_contour, ref_energy_contour)
                            p_diffs.append(p_diff)
                            e_diffs.append(e_diff)
                        p_diffs_mean = mean_func(p_diffs)
                        e_diffs_mean = mean_func(e_diffs)
                        psd_mcd_stat_res[spk][emo][model_n][fine_categ].append(p_diffs_mean)
                        psd_mcd_stat_res[spk][emo][model_n][fine_categ].append(e_diffs_mean)
    return psd_mcd_stat_res


def statcz_psd_mcd(prosody_dict, exclude_zero=False):
    """
    compute psd ctw, mcd, and staticize by mean over each (spk)/emo/model
    Args:
        prosody_dict:  {"spk": {"emo1": {"model1": {"psd/phone/speechid": [[], []]}}}}
    Returns:
        psd_ctw_res: {"spk": "emo1": {"model1": [[wav1_pitch_ctw, wav2_pitch_ctw, ...], [wav1_eng_ctw, wav2_eng_ctw, ...]]}}
        psd_mcd_stat_res: {"spk": "emo1": {"model1": [[p_diff], [e_diff], [mcd_res]]}}
        result: {"emo1": {"model1": [[p_diff], [e_diff], [mcd_res]]}}
    """
    psd_mcd_stat_res = {}
    psd_ctw_res = {}
    mean_func = lambda x: np.mean(np.array(x))

    for spk, emo_model_psd in prosody_dict.items():
        if spk not in psd_mcd_stat_res.keys():
            psd_mcd_stat_res[spk], psd_ctw_res[spk] = dict(), dict()
        for emo, model_psd in emo_model_psd.items():
            if emo not in psd_mcd_stat_res[spk].keys():
                psd_mcd_stat_res[spk][emo], psd_ctw_res[spk][emo] = dict(), dict()
            for model_n, psd_phone_sid in model_psd.items():
                if model_n != "reference":
                    if model_n not in psd_mcd_stat_res[spk][emo].keys():
                        psd_mcd_stat_res[spk][emo][model_n], psd_ctw_res[spk][emo][model_n] = [], []
                    wav_num = len(psd_phone_sid["pitch"])
                    p_diffs = []
                    e_diffs = []
                    for i in range(wav_num):  # reference id not in order (model["pitch"]["i"] is not referenced from a["reference"]["pitch"]["i"])
                        # get reference id
                        ref_i = psd_phone_sid["speechid"][i].split("ref")[1].split("_")[0]
                        #ref_i = int(psd_phone_sid["speechid"][i].split("_")[-2][-1])
                        # get reference index of which serve as reference to speech i
                        ref_i_index = [i for i, v in enumerate(prosody_dict[spk][emo]["reference"]["speechid"]) if f"ref{ref_i}" in v]
                        assert len(ref_i_index) == 1
                        ref_i_index = ref_i_index[0]

                        # deal with nan and 0
                        syn_pitch_contour, ref_pitch_contour = (interpolate_nan(psd_phone_sid["pitch"][i]),
                                                                interpolate_nan(prosody_dict[spk][emo]["reference"]["pitch"][ref_i_index]))
                        syn_energy_contour, ref_energy_contour = (interpolate_nan(psd_phone_sid["energy"][i]),
                                                                  interpolate_nan(prosody_dict[spk][emo]["reference"]["energy"][ref_i_index]))
                        if exclude_zero:
                            syn_pitch_contour, syn_energy_contour = interpolate_unvoiced(syn_pitch_contour, syn_energy_contour, rm_approach=False)
                            ref_pitch_contour, ref_energy_contour = interpolate_unvoiced(ref_pitch_contour, ref_energy_contour, rm_approach=False)
                        #print("pitch diff", syn_pitch_contour[:30], ref_pitch_contour[:30])
                        p_diff, _ = dtw_sim_score(syn_pitch_contour, ref_pitch_contour)
                        e_diff, _ = dtw_sim_score(syn_energy_contour, ref_energy_contour)
                        p_diffs.append(p_diff)
                        e_diffs.append(e_diff)
                    p_diffs_mean = mean_func(p_diffs)
                    e_diffs_mean = mean_func(e_diffs)
                    psd_ctw_res[spk][emo][model_n].append(p_diffs)
                    psd_ctw_res[spk][emo][model_n].append(e_diffs)
                    psd_ctw_res[spk][emo][model_n].append(psd_phone_sid["speechid"])
                    psd_mcd_stat_res[spk][emo][model_n].append(p_diffs_mean)
                    psd_mcd_stat_res[spk][emo][model_n].append(e_diffs_mean)

    # compute mean
    collector = {}
    for spk_data in psd_mcd_stat_res.values():
        for emotion, models in spk_data.items():
            if emotion not in collector:
                collector[emotion] = {m: [] for m in models}
            for model, values in models.items():
                collector[emotion][model].append(values)
    # Compute means over speaker
    result = {}
    for emotion, models in collector.items():
        result[emotion] = {}
        for model, values in models.items():
            averaged = [mean(col) for col in zip(*values)]
            result[emotion][model] = averaged
    return psd_ctw_res, psd_mcd_stat_res, result


def interpolate_unvoiced(pitch, energy, rm_approach):
    """
    Interpolate unvoiced regions (pitch == 0) in both pitch and energy contours.

    Args:
        pitch (array-like): pitch contour (Hz), zeros = unvoiced
        energy (array-like): energy contour aligned with pitch

    Returns:
        pitch_interp (np.ndarray): pitch with interpolated unvoiced parts
        energy_interp (np.ndarray): energy with interpolated unvoiced parts
    """
    # -------------------------------
    # 1) Match energy length to pitch
    # -------------------------------
    if len(energy) < len(pitch):
        # pad energy using last value
        pad_len = len(pitch) - len(energy)
        energy = np.pad(energy, (0, pad_len), mode='edge')
    elif len(energy) > len(pitch):
        # cut extra frames
        energy = energy[:len(pitch)]

    if rm_approach:
        pitch_contour = pitch[pitch > 0]
        energy_contour = energy[energy > 0]
        return pitch_contour, energy_contour

    pitch = np.asarray(pitch, dtype=float)
    energy = np.asarray(energy, dtype=float)

    # indices where pitch is valid (voiced)
    valid_idx = np.where(pitch != 0)[0]
    # indices where pitch is unvoiced (zero)
    uv_idx = np.where(pitch == 0)[0]

    # if no voiced frames exist, return original
    if len(valid_idx) == 0:
        return pitch, energy

    # interpolate pitch
    pitch_interp = pitch.copy()
    pitch_interp[uv_idx] = np.interp(uv_idx, valid_idx, pitch[valid_idx])

    # interpolate energy at the SAME positions
    energy_interp = energy.copy()
    energy_interp[uv_idx] = np.interp(uv_idx, valid_idx, energy[valid_idx])

    return pitch_interp, energy_interp


def extract_psd_fine_class(prosody_dict, base_dir):
    """

    For parallel test
    https://medium.com/@markstent/dynamic-time-warping-a8c5027defb6
    Args:
        prosody_dict:  {"spk": {"emo1": {"model1": {"psd/phone/speechid": [[], []]}}}}


    Returns:
        psd_mcd_stat_res: {"spk": "emo1": {"model1": [[p_diff], [e_diff], [mcd_res]]}}}}
    """
    psd_mcd_stat_res = {}

    #models_name = [model_n for model_n in prosody_dict["Angry"].keys().tolist() if model_n != "reference"]
    diff_func = lambda x, y: np.mean(np.abs(np.array(x) - np.array(y)))
    mean_func = lambda x: np.mean(np.array(x))

    for spk, emo_model_psd in prosody_dict.items():
        if spk not in psd_mcd_stat_res.keys():
            psd_mcd_stat_res[spk] = dict()
        for emo, model_psd in emo_model_psd.items():
            if emo not in psd_mcd_stat_res[spk].keys():
                psd_mcd_stat_res[spk][emo] = dict()
            for model_n, psd_phone_sid in model_psd.items():
                if model_n != "reference":
                    if model_n not in psd_mcd_stat_res[spk][emo].keys():
                        psd_mcd_stat_res[spk][emo][model_n] = []
                    wav_num = len(psd_phone_sid["pitch"])
                    p_diffs = []
                    e_diffs = []
                    for i in range(wav_num):
                        syn_pitch_contour = interpolate_nan(psd_phone_sid["pitch"][i])
                        ref_pitch_contour = interpolate_nan(prosody_dict[spk][emo]["reference"]["pitch"][i])
                        syn_energy_contour = interpolate_nan(psd_phone_sid["energy"][i])
                        ref_energy_contour = interpolate_nan(prosody_dict[spk][emo]["reference"]["energy"][i])
                        p_diff, _ = dtw_sim_score(syn_pitch_contour, ref_pitch_contour)
                        e_diff, _ = dtw_sim_score(syn_energy_contour, ref_energy_contour)
                        p_diffs.append(p_diff)
                        e_diffs.append(e_diff)
                    p_diffs_mean = mean_func(p_diffs)
                    e_diffs_mean = mean_func(e_diffs)
                    psd_mcd_stat_res[spk][emo][model_n].append(p_diffs_mean)
                    psd_mcd_stat_res[spk][emo][model_n].append(e_diffs_mean)

                    # statcz mcd
                    wav_dir = os.path.join(base_dir, model_n)
                    ref_dir = os.path.join(base_dir, "reference")
                    mcds = [
                        mcd_toolbox.calculate_mcd(wav_dir + "/" + psd_phone_sid["speechid"][i] + ".wav",
                                                  ref_dir + "/" + prosody_dict[spk][emo]["reference"]["speechid"][i]  + ".wav")
                        for i in range(wav_num)]
                    mcd_res_mean = mean_func(mcds)
                    psd_mcd_stat_res[spk][emo][model_n].append(mcd_res_mean)
    return psd_mcd_stat_res


def dtw_sim_score(time_series_a, time_series_b):
    distance, paths = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
    best_path = dtw.best_path(paths)
    #print("distance:", distance)
    #print("paths and its len:", paths, paths.shape)
    #print("best_path and its length", best_path, len(best_path))
    similarity_score = distance / len(best_path)
    return similarity_score, best_path


if __name__ == '__main__':
    a = np.array([0, 0, 1, 2])
    ain = interpolate_nan(a)
    #print(a[a>0])

    a = [131.61519793218153,
        137.773230681949,
        186.03541642425648,
        220.50840624019034,
        254.98139605612425,
        358.4003655039259,
        481.29177186936136,
        393.18286814331304,
        294.14826773218607,
        207.49299237245,
        139.4067045898002,
        123.36547599749247]

    b = [112.48888479288458,
        112.48888479288458,
        112.48888479288458,
        111.85515790193942,
        105.67122734175119,
        115.76995832186587,
        115.13092596656963,
        133.87821462160488,
        124.64575470269426,
        114.2883945637247,
        140.52005456170232,
        133.8224072106353,
        118.19863178251124,
        143.38982009143623]

    from test_data import a1, b1
    from vis2 import draw_dtw

    a1 = np.array(a1)
    b1 = np.array(b1)
    a1 = interpolate_nan(a1)
    b1 = interpolate_nan(b1)
    a1 = a1[a1 > 0]
    b1 = b1[b1 > 0]
    #ab_dist = dtw_sim_score(a1, b1)
    #print(len(a1), len(b1))
    #print(ab_dist)

    ab_dist, best_path = dtw_sim_score(a1, b1)
    print(ab_dist)
    draw_dtw(a1, b1, best_path, output_png="res/dtw_spk19_ang_ref4_syn0.png")
