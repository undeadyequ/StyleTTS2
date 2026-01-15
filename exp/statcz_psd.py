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
    # compute mean over speaker
    collector = {}
    for spk_data in psd_mcd_stat_res.values():
        for emotion, models in spk_data.items():
            if emotion not in collector:
                collector[emotion] = {m: [] for m in models}
            for model, values in models.items():
                collector[emotion][model].append(values)
    result = {}
    for emotion, models in collector.items():
        result[emotion] = {}
        for model, values in models.items():
            averaged = [mean(col) for col in zip(*values)]
            result[emotion][model] = averaged
    return psd_ctw_res, psd_mcd_stat_res, result


def statcz_psd_mcd_fine_class2(prosody_dict, exclude_zero=False):
    """
    compute psd ctw, mcd, and staticize by mean over each (spk)/emo/model
    Args:
        prosody_dict:  {"spk": {"emo1": {"model1": {"psd/phone/speechid": [[], []]}}}}
    Returns:
        psd_ctw_res: {"spk": "emo1": {"model1": [[wav1_pitch_ctw, wav2_pitch_ctw, ...], [wav1_eng_ctw, wav2_eng_ctw, ...]]}}
        psd_mcd_stat_res: {"spk": "emo1": {"model1": [[p_diff], [e_diff], [mcd_res]]}}
        result: {"emo1": {"model1": [[p_diff], [e_diff], [mcd_res]]}}
    """
    # OUT
    psd_mcd_stat_res = {}
    psd_ctw_res = {}

    mean_func = lambda x: np.mean(np.array(x))

    for spk, emo_model_psd in prosody_dict.items():
        if spk not in psd_mcd_stat_res.keys():
            psd_mcd_stat_res[spk], psd_ctw_res[spk] = dict(), dict()
        for emo, model_psd in emo_model_psd.items():
            if emo not in psd_mcd_stat_res[spk].keys():
                psd_mcd_stat_res[spk][emo], psd_ctw_res[spk][emo] = dict(), dict()
            for model_n, fine_psd_phone_sid in model_psd.items():
                if model_n != "reference":
                    if model_n not in psd_mcd_stat_res[spk][emo].keys():
                        psd_mcd_stat_res[spk][emo][model_n], psd_ctw_res[spk][emo][model_n] = dict(), dict()
                        for fine_cate, psd_phone_sid in fine_psd_phone_sid.items():
                            psd_mcd_stat_res[spk][emo][model_n][fine_cate], psd_ctw_res[spk][emo][model_n][fine_cate] = [], []

                            wav_num = len(psd_phone_sid["pitch"])
                            p_diffs = []
                            e_diffs = []
                            for i in range(wav_num):  # reference id not in order (model["pitch"]["i"] is not referenced from a["reference"]["pitch"]["i"])
                                # get reference id
                                ref_i = psd_phone_sid["speechid"][i].split("ref")[1].split("_")[0]

                                ref_i_index = [i for i, v in enumerate(prosody_dict[spk][emo]["reference"][fine_cate]["speechid"]) if f"ref{ref_i}" in v]
                                assert len(ref_i_index) == 1
                                ref_i_index = ref_i_index[0]

                                # deal with nan and 0
                                syn_pitch_contour, ref_pitch_contour = (interpolate_nan(psd_phone_sid["pitch"][i]),
                                                                        interpolate_nan(prosody_dict[spk][emo]["reference"][fine_cate]["pitch"][ref_i_index]))
                                syn_energy_contour, ref_energy_contour = (interpolate_nan(psd_phone_sid["energy"][i]),
                                                                          interpolate_nan(prosody_dict[spk][emo]["reference"][fine_cate]["energy"][ref_i_index]))
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
                            psd_ctw_res[spk][emo][model_n][fine_cate].append(p_diffs)
                            psd_ctw_res[spk][emo][model_n][fine_cate].append(e_diffs)

                            psd_ctw_res[spk][emo][model_n][fine_cate].append(psd_phone_sid["speechid"])
                            psd_mcd_stat_res[spk][emo][model_n][fine_cate].append(p_diffs_mean)
                            psd_mcd_stat_res[spk][emo][model_n][fine_cate].append(e_diffs_mean)

    collector = {}
    # Collect stats
    for spk_data in psd_mcd_stat_res.values():
        for emotion, emo_data in spk_data.items():
            collector.setdefault(emotion, {})
            for model, model_data in emo_data.items():
                collector[emotion].setdefault(model, {})
                for datatype, pe_list in model_data.items():
                    collector[emotion][model].setdefault(datatype, [])
                    collector[emotion][model][datatype].append(pe_list)
    # Compute mean
    output_dict = {}
    for emotion, emo_data in collector.items():
        output_dict[emotion] = {}
        for model, model_data in emo_data.items():
            output_dict[emotion][model] = {}
            for datatype, values in model_data.items():
                values = np.asarray(values)  # (num_speakers, 2)
                output_dict[emotion][model][datatype] = values.mean(axis=0).tolist()

    return psd_ctw_res, psd_mcd_stat_res, output_dict


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

    a = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        188.52698314694513,
                        191.62290276915766,
                        181.85544998980848,
                        160.43214298580475,
                        191.98927560744121,
                        191.64906822340197,
                        181.28408304456562,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        193.06533580138196,
                        190.21837232753595,
                        189.43077781980136,
                        189.07676352469647,
                        189.97062534345255,
                        189.73388702618533,
                        189.2507303071056,
                        186.90828693958747,
                        183.88816604660192,
                        183.557928414151,
                        177.54644032916605,
                        173.6090452118843,
                        154.564724614881,
                        141.1546223870995,
                        132.52714165178512,
                        133.6253786080943,
                        132.42905641920288,
                        140.52981677740289,
                        148.28528597458188,
                        158.16831221183324,
                        162.74956427312776,
                        164.58627589181827,
                        167.82644718146824,
                        171.642575918701,
                        171.74883747760651,
                        171.69082034798933,
                        174.36087330102313,
                        180.98599521683195,
                        191.96482884961358,
                        190.14048119038964,
                        184.43024926563984,
                        172.08693081863083,
                        0.0,
                        146.30802449161212,
                        149.75055988270927,
                        150.90479524343144,
                        158.80252895978393,
                        164.992785588315,
                        168.33851578755937,
                        169.56768769278196,
                        168.45946887166298,
                        165.3350076369196,
                        159.16687489323348,
                        151.89757272610387,
                        146.17796957804475,
                        0.0,
                        0.0,
                        0.0,
                        237.1046348741109,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        150.16294177550097,
                        148.31565578686343,
                        131.2881182348143,
                        119.52386430319099,
                        117.60304630192607,
                        118.9961480085441,
                        121.65838360049635,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        293.692183477926,
                        188.98875417399074,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        152.95500602310617,
                        159.76679868498283,
                        164.07915731539498,
                        167.34383560534926,
                        171.82068747235797,
                        176.82709528978359,
                        181.26463868212252,
                        189.592221691578,
                        197.98302315301848,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        142.38152159074326,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        157.44496847372062,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        189.06684960167308,
                        194.84660622525266,
                        196.9453265276997,
                        197.4206532195516,
                        196.87378759494453,
                        193.80902085025966,
                        187.53692096559752,
                        170.9718547283815,
                        192.42661632430776,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        182.83557055302356,
                        165.64761302065784,
                        166.48483232475277,
                        169.7705287670305,
                        171.5141528869464,
                        173.63734835916725,
                        175.9558029384665,
                        177.8112427922909,
                        181.47177344325485,
                        199.88685279640316,
                        194.85096326459347,
                        191.58366482954924,
                        185.6009297024174,
                        171.8869344021593,
                        157.35378592405917,
                        147.17932742133513,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]

    b = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        188.55275305149945,
                        184.16261069338267,
                        194.63119606075267,
                        200.30979131345146,
                        212.21655695661016,
                        213.62646174358002,
                        212.86888128985956,
                        200.5236884450991,
                        193.6318312089317,
                        201.6041521469804,
                        212.80197251387355,
                        200.46671604858508,
                        198.6599410360136,
                        191.15877480562077,
                        179.83556556546523,
                        162.7035545482115,
                        154.46625830008637,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        242.0378490602336,
                        245.56871164540422,
                        234.21300630730263,
                        215.54516834662155,
                        215.75048044258682,
                        235.2042922357766,
                        253.97250717650573,
                        259.46446502774467,
                        262.29800588567,
                        266.79189863257096,
                        272.3398191067238,
                        277.52754721664274,
                        282.4243389564452,
                        284.4154662466119,
                        289.64213945029445,
                        291.0594364695582,
                        291.1750590919686,
                        288.7281658492073,
                        288.65945587736485,
                        289.31180981005895,
                        286.74499100016317,
                        280.64747703473205,
                        271.968890279283,
                        256.91749494837103,
                        242.5712658371653,
                        222.69267614371853,
                        196.9286054280976,
                        179.65710356425492,
                        172.49439396206338,
                        171.4292188846453,
                        163.62784566128082,
                        158.44996302227221,
                        146.12952877987962,
                        0.0,
                        114.62243225822809,
                        100.06233175038923,
                        101.0349303589402,
                        98.34881872859908,
                        111.47834150773755,
                        0.0,
                        0.0,
                        0.0,
                        150.47573715591332,
                        165.02205242299465,
                        148.32913588066205,
                        147.2061167364509,
                        154.46493547011283,
                        153.82833738752134,
                        150.57626819373243,
                        146.84158448776614,
                        143.10559744740308,
                        138.96330020452393,
                        136.02967259335037,
                        133.0218676905473,
                        129.91112591180072,
                        126.62707262281519,
                        123.26368925513654,
                        120.43087947359639,
                        118.77254576230396,
                        115.76104638824087,
                        113.14198374177896,
                        110.11872112267005,
                        104.12630810216848,
                        103.5837825104753,
                        96.31533995708239,
                        95.89707918756407,
                        92.00315329461941,
                        88.10195393722455,
                        88.0489764963118,
                        88.48639178743255,
                        93.46317095573305,
                        99.73670905717579,
                        98.70467853710907,
                        99.26766097028134,
                        100.66585114670387,
                        96.81508287568714,
                        114.62096234160595,
                        93.30182923457946,
                        104.9492191867891,
                        110.7665488241482,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]

    from test_data import a1, b1
    from vis2 import draw_dtw

    a1 = np.array(a)
    b1 = np.array(b)
    a1 = interpolate_nan(a1)
    b1 = interpolate_nan(b1)
    a1 = a1[a1 > 0]
    b1 = b1[b1 > 0]
    #ab_dist = dtw_sim_score(a1, b1)
    print(a1, b1)
    #print(ab_dist)

    ab_dist, best_path = dtw_sim_score(a1, b1)
    print(ab_dist)
    #draw_dtw(a1, b1, best_path, output_png="res/dtw_spk19_ang_ref4_syn0.png")


    #from exp.main_piolot_test import normalized_dtw_1d
    #dtw_res2 = normalized_dtw_1d(a1, b1)
    #print(dtw_res2)
