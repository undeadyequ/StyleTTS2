"""
Main pictures
0. pitch/energy contours comparison over models.

1. band_matrix and monoGuided attentions (Row 2 * col 4)
    - sigma = 0, 0.2, 0.5, 0.8
2. monoGuided attentions and reference/synthesized mel
    - find good sample on sigma=0, sigma=0.5 -> good: local style perfectly transferred from ref -> syn when sigma=0.5 (not good when sigma=0.0)
    - get their mel, attn
3. conditioning pitch types and pitch of synthesized speech conditioned on these
    - find good sample on ref_pe, pred_pe, fuse_pe (fused_pe of synthesizsed near to ref_pe)

Other Pictures
"""
import json
import os.path
from exp.ref_aware_pe3 import fuse_prosody_smooth_additive
from vis_data_adaptor import phone2syl
import matplotlib.pyplot as plt

from visualization import vis_mono_guide_mask
from vis2 import plot_mel_with_pitch, plot_attn_with_rect, plot_lines, plot_mel_with_pitch2, plot_attn_with_rect2
from inference_1_or_2 import inference_second, get_second_model
import numpy as np
from utilities.guide_mask import make_guided_attention_masks2
from visualization import vis_matrix_attn, vis_matrix_attn2
from itertools import accumulate

model_config = {
        "mdit_cfm_v10": ["first_txt2mel_cfm_v10/epoch_2nd_00048.pth",
                        "first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml"]}

inference_args = {
        "alpha": 0.8,
        "beta": 0.8,
        "diffusion_steps": 10,
        "embedding_scale": 1,
        "style_dim": 256,
        "mix_ref_pe_type": "ref_pred_add",  # ref_pred_add none ref_pred_gate
        "cfg_strength": 3,
        "mono_guide_delta": 0.3,
        "Vis_F0": False,
        "fuse_beta": 0.3
    }

word2phone = lambda w : [c for c in ' '.join(w)]

def read_attn(attn_path, show_t=0, show_b=5, show_h=0):
    attn = np.load(attn_path, allow_pickle=True)
    if len(attn.shape) == 6:
        attn = attn[show_t, show_b, 0, show_h, ...]  # [time, block_n, batch, head, t_t, t_s]
    elif len(attn.shape) == 5:
        attn = attn[show_b, 0, show_h, ...]  # [block_n, batch, head, t_t, t_s]  # No batch
    else:
        attn = attn[show_b, show_h, ...]  # [block_n, head, t_t, t_s]    # in synthesize_from_batch, batch dim is removed
    return attn


def draw_psd_contour(img_out):
    psd_compare_png = os.path.join(img_out, f"psd_{cmp_modelnames_combine}_ref{show_text[0]}_syn{show_text[1]}.png")
    pitch_cmp_path, energy_cmp_path = (os.path.join(out_dir, f"vis_pitch_{cmp_modelnames_combine}.json"),
                                       os.path.join(out_dir, f"vis_energy_{cmp_modelnames_combine}.json"))
    # visualize psd
    pitch_dict_for_vis, energy_dict_for_vis = convert_vis_psd_json(
        psd_cmp_dict["spk" + show_spk], show_text, cutpad_reference=False,
        save_dict=(pitch_cmp_path, energy_cmp_path))  # for_vis: {"emo1": {"model1": list(psd_len)}}}
    vis_psd(pitch_dict_for_vis, energy_dict_for_vis, psd_compare_png, show_text, ordered_lengend=orderd_cmp_modelnames)


def create_bandMatrix_monoGuideAttns(speech_id, band_sigmas, show_t=0, show_b=0, show_h=0):
    mono_ablation_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono/monoDiT_ablation"
    mono_sigma_attn_dirs = ["none_m02_f02_attn", "none_m05_f02_attn", "none_m08_f02_attn", "none_mm10_f02_attn"]
    bandMatrixs = []
    monoGuideAttns = []
    for attn_dir in mono_sigma_attn_dirs:
        attn_path = os.path.join(mono_ablation_dir, attn_dir, speech_id + ".npy")
        attn = read_attn(attn_path, show_t, show_b, show_h)
        monoGuideAttns.append(attn)
    for sigma in band_sigmas:
        ilens, olens = [monoGuideAttns[0].shape[-2]], [monoGuideAttns[0].shape[-1]]
        guide_matrix = 1 - make_guided_attention_masks2(ilens, olens, base_sigma=sigma, eps=0.002).cpu().numpy()
        if guide_matrix.mean() == 1:   # set all-one matrix to yellow, instead of blue
            guide_matrix[0, 0, 0] = 0
        bandMatrixs.append(guide_matrix.squeeze())
    return bandMatrixs, monoGuideAttns

def draw_bandMatrix_monoGuideAttns(speech_id, band_sigmas, show_t=0, show_b=0, show_h=0, out_png="band_matrix_mono_attn.png"):
    """ draw bands and attns when generated speech_id, conditioned on differentband_sigmas
        band1 band2 band3
        attn1 attn2 attn3
    """
    bandMatrixs, monoGuideAttns = create_bandMatrix_monoGuideAttns(speech_id, band_sigmas, show_t=show_t, show_b=show_b, show_h=show_h)
    vis_matrix_attn2(bandMatrixs, monoGuideAttns, out_png)

def set_academic_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.linewidth": 0.6,
        "figure.dpi": 350,
        "savefig.dpi": 600,
        "text.usetex": False,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0
    })


def draw_monGuideAttn_refSynMel(speech_id="spk0019_Surprise_ref3_syn0", band_sigmas=("none_m02_f02", "none_m10_f02"),
                                out_png="monGuideAttn_refSynMel.png", show_t=0, show_b=0, show_h=0):
    """
    mel_ref  attn_dit attn_mdit
             mel_dit  mel_mdit
    """
    set_academic_style()

    mono_ablation_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono/monoDiT_ablation"
    ref_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono/reference/random"
    attn_path = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono/attn_mdit_random.json"

    show_spk, show_emo = speech_id.split("_")[:2]
    ref_id = speech_id.rsplit("_", 1)[0]
    #ref_id = "spk0019_Surprise_ref3"

    # Styling presets
    attn_kwargs = {
        "fontsize": 11,
        "x_rotation": 45, "y_rotation": 45, "yticklabel_rotation": 90,
        "xy_ticklabs_bold_index": ([4], [4]),
        "xy_auxline": (None, None),
        "xy_rectangle": None,
        "xy_rectangle_bold_index": [2]
    }
    speech_kwargs = {
        "fontsize": 11,
        "x_rotation": 45, "y_rotation": 45,
        "xy_ticklabs_bold_index": ([4], [4]),
        "xy_rectangle": None,
        "xy_rectangle_bold_index": [2]
    }
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.subplots_adjust(wspace=0.25, hspace=0.30)

    # Model names
    monoGuided_model, origin_model = band_sigmas

    # attn related info
    with open(attn_path, "r") as f:
        attn_dict = json.load(f)

    # 1. Image content: mel and attn
    ## mel of ref and syn
    wav_dirs = [f"{monoGuided_model}", f"{origin_model}"]
    origin_wav = os.path.join(mono_ablation_dir, wav_dirs[1], speech_id + ".wav")
    monoGuided_wav = os.path.join(mono_ablation_dir, wav_dirs[0], speech_id + ".wav")
    ref_wav = os.path.join(ref_dir, ref_id + ".wav")

    ## attn
    mono_sigma_attn_dirs = [f"{monoGuided_model}_attn", f"{origin_model}_attn"]
    origin_attn = os.path.join(mono_ablation_dir, mono_sigma_attn_dirs[1], speech_id + ".npy")
    origin_attn = read_attn(origin_attn, show_t, show_b, show_h)
    monoGuideAttn = os.path.join(mono_ablation_dir, mono_sigma_attn_dirs[0], speech_id + ".npy")
    monoGuideAttn = read_attn(monoGuideAttn, show_t, show_b, show_h)

    # 2. Image coordinate info: phone/dur of ref and syn
    ref_txt_index = attn_dict[show_spk][show_emo][origin_model]["speechid"].index(speech_id)
    origin_syn_phones, origin_syn_durs = (attn_dict[show_spk][show_emo][origin_model]["syn_phonemes"][ref_txt_index],
                                          attn_dict[show_spk][show_emo][origin_model]["q_dur"][ref_txt_index])
    monoGuide_syn_phones, monoGuide_syn_durs = (attn_dict[show_spk][show_emo][monoGuided_model]["syn_phonemes"][ref_txt_index],
                                          attn_dict[show_spk][show_emo][monoGuided_model]["q_dur"][ref_txt_index])
    ref_phones, ref_durs = (attn_dict[show_spk][show_emo][origin_model]["ref_phonemes"][ref_txt_index],
                                          attn_dict[show_spk][show_emo][origin_model]["k_dur"][ref_txt_index])

    # 3. draw
    ## convert x/y ticks, ticklabs
    USE_SYLLABLE = True
    if USE_SYLLABLE:
        #syn_clean_syl_start_index = [0, 4, 7, 10, 15, 18]    # Add-hoc
        #ref_clean_syl_start_index = [0, 3, 7, 12, 15, 19, 21]

        syn_clean_syl_start_index = [0, 4, 8, 12, 14, 19, 22]  # Add-hoc
        ref_clean_syl_start_index = [0, 3, 5, 7, 11, 13, 18, 20, 24, 26]

        # split phone of word to phones
        origin_syn_phones = word2phone(origin_syn_phones)
        origin_syn_phones.insert(0, "")
        monoGuide_syn_phones = word2phone(monoGuide_syn_phones)
        monoGuide_syn_phones.insert(0, "")
        ref_phones = word2phone(ref_phones)
        ref_phones.insert(0, "")

        # * 2
        origin_syn_durs = np.array(origin_syn_durs) * 2
        monoGuide_syn_durs = np.array(monoGuide_syn_durs) * 2
        ref_durs = np.array(ref_durs) * 2

        xticks, x_ticklabs = phone2syl(origin_syn_phones, origin_syn_durs, syl_start_index=syn_clean_syl_start_index)
        xticks_mono, x_ticklabs_mono = phone2syl(monoGuide_syn_phones, monoGuide_syn_durs, syl_start_index=syn_clean_syl_start_index)
        yticks, y_ticklabs = phone2syl(ref_phones, ref_durs, syl_start_index=ref_clean_syl_start_index)

    else:
        xticks = np.array(list(accumulate(origin_syn_durs))) * 2
        xticks_mono = np.array(list(accumulate(monoGuide_syn_durs))) * 2
        yticks = np.array(list(accumulate(ref_durs))) * 2

        x_ticklabs = word2phone(origin_syn_phones)
        x_ticklabs_mono = word2phone(monoGuide_syn_phones)
        y_ticklabs = word2phone(ref_phones)

    # set frame(origin_wav) = frame(monoGuide_wav) = min(origin_wav, monoGuide_wav), including xticks_*, *_attn <- they are diff because of denoising initial noise
    min_frame = int(min(xticks[-1], xticks_mono[-1]))
    xticks[-1], xticks_mono[-1] = min_frame, min_frame
    origin_attn = origin_attn[:min_frame, :min_frame]
    monoGuideAttn = monoGuideAttn[:min_frame, :min_frame]

    # set frame(referent_wav) = min(origin_wav, monoGuide_wav), including yticks
    scaled_yticks = yticks / yticks[-1] * min_frame

    ## 1st row:
    plot_mel_with_pitch(axs[0, 0], "Reference", ref_wav, ("", scaled_yticks, y_ticklabs), speech_kwargs)   # subtitle, attn, xlabel/xticks/xticklab, ylabel/yticks/yticklab, kwargs
    plot_attn_with_rect(axs[0, 1], r"$\delta = \varnothing$", origin_attn, ("", xticks, x_ticklabs), ("", scaled_yticks, y_ticklabs), attn_kwargs)
    plot_attn_with_rect(axs[0, 2], r"$\delta = 0.5$", monoGuideAttn, ("", xticks_mono, x_ticklabs_mono), ("", scaled_yticks, y_ticklabs), attn_kwargs)

    ## 2nd row
    axs[1, 0].axis('off')
    plot_mel_with_pitch(axs[1, 1], "", origin_wav, ("", xticks, x_ticklabs), speech_kwargs, max_len=min_frame)
    plot_mel_with_pitch(axs[1, 2], "", monoGuided_wav, ("", xticks_mono, x_ticklabs_mono), speech_kwargs, max_len=min_frame)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def draw_monGuideAttn_refSynMel2(
        speech_id="spk0019_Surprise_ref3_syn0",
        band_sigmas=("none_m02_f02", "none_m10_f02"),
        out_png="monGuideAttn_refSynMel.png",
        show_t=0, show_b=0, show_h=0):

    set_academic_style()  # <<< NEW

    # Directories
    mono_ablation_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono/monoDiT_ablation"
    ref_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono/reference/random"
    attn_path = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono/attn_mdit_random.json"

    show_spk, show_emo = speech_id.split("_")[:2]
    ref_id = speech_id.rsplit("_", 1)[0]

    # Styling presets
    attn_kwargs = {
        "fontsize": 11,
        "x_rotation": 30, "y_rotation": 30, "yticklabel_rotation": 90,
        "xy_ticklabs_bold_index": ([4], [4]),
        "xy_auxline": (None, None),
        "xy_rectangle": None,
    }
    speech_kwargs = {
        "fontsize": 11,
        "x_rotation": 45, "y_rotation": 30,
        "xy_ticklabs_bold_index": ([4], [4]),
        "xy_rectangle": None,
    }

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    fig.subplots_adjust(wspace=0.25, hspace=0.30)

    # Load attention json
    with open(attn_path, "r") as f:
        attn_dict = json.load(f)

    # Model names
    monoGuided_model, origin_model = band_sigmas

    # Load wav files
    origin_wav = os.path.join(mono_ablation_dir, origin_model, speech_id + ".wav")
    monoGuided_wav = os.path.join(mono_ablation_dir, monoGuided_model, speech_id + ".wav")
    ref_wav = os.path.join(ref_dir, ref_id + ".wav")

    # Load attentions
    origin_attn = read_attn(os.path.join(mono_ablation_dir, origin_model + "_attn", speech_id + ".npy"),
                            show_t, show_b, show_h)
    monoGuideAttn = read_attn(os.path.join(mono_ablation_dir, monoGuided_model + "_attn", speech_id + ".npy"),
                              show_t, show_b, show_h)

    # Extract phone/duration info from dictionary
    idx = attn_dict[show_spk][show_emo][origin_model]["speechid"].index(speech_id)

    origin_syn_phones = attn_dict[show_spk][show_emo][origin_model]["syn_phonemes"][idx]
    origin_syn_durs = attn_dict[show_spk][show_emo][origin_model]["q_dur"][idx]

    monoGuide_syn_phones = attn_dict[show_spk][show_emo][monoGuided_model]["syn_phonemes"][idx]
    monoGuide_syn_durs = attn_dict[show_spk][show_emo][monoGuided_model]["q_dur"][idx]

    ref_phones = attn_dict[show_spk][show_emo][origin_model]["ref_phonemes"][idx]
    ref_durs = attn_dict[show_spk][show_emo][origin_model]["k_dur"][idx]

    # === Syllable-level preprocessing (unchanged logic) ===
    USE_SYLLABLE = True
    if USE_SYLLABLE:
        syn_clean_syl_start_index = [0, 4, 8, 12, 14, 19, 22]
        ref_clean_syl_start_index = [0, 3, 7, 11, 13, 18, 20, 24, 26]

        # Convert word→phone and add blank
        origin_syn_phones = [""] + word2phone(origin_syn_phones)
        monoGuide_syn_phones = [""] + word2phone(monoGuide_syn_phones)
        ref_phones = [""] + word2phone(ref_phones)

        origin_syn_durs = np.array(origin_syn_durs) * 2
        monoGuide_syn_durs = np.array(monoGuide_syn_durs) * 2
        ref_durs = np.array(ref_durs) * 2

        xticks, x_ticklabs = phone2syl(origin_syn_phones, origin_syn_durs, syl_start_index=syn_clean_syl_start_index)
        xticks_mono, x_ticklabs_mono = phone2syl(monoGuide_syn_phones, monoGuide_syn_durs, syl_start_index=syn_clean_syl_start_index)
        yticks, y_ticklabs = phone2syl(ref_phones, ref_durs, syl_start_index=ref_clean_syl_start_index)
    else:
        # fallback (unchanged)
        ...

    # === Align lengths (your logic retained) ===
    min_frame = int(min(xticks[-1], xticks_mono[-1]))
    xticks[-1], xticks_mono[-1] = min_frame, min_frame

    origin_attn = origin_attn[:min_frame, :min_frame]
    monoGuideAttn = monoGuideAttn[:min_frame, :min_frame]

    scaled_yticks = yticks / yticks[-1] * min_frame

    # === ROW 1: Reference + attention maps ===
    plot_mel_with_pitch2(axs[0, 0], "Reference Speech", ref_wav,
                        ("", scaled_yticks, y_ticklabs), speech_kwargs)

    plot_attn_with_rect2(
        axs[0, 1], r"$\delta = \varnothing$",
        origin_attn,
        ("", xticks, x_ticklabs),
        ("", scaled_yticks, y_ticklabs),
        attn_kwargs
    )

    plot_attn_with_rect2(
        axs[0, 2], r"$\delta = 0.5$",
        monoGuideAttn,
        ("", xticks_mono, x_ticklabs_mono),
        ("", scaled_yticks, y_ticklabs),
        attn_kwargs
    )

    # === ROW 2: synthesized mels ===
    axs[1, 0].axis("off")

    plot_mel_with_pitch2(axs[1, 1], "Baseline Synthesis", origin_wav,
                        ("", xticks, x_ticklabs), speech_kwargs,
                        max_len=min_frame)

    plot_mel_with_pitch2(axs[1, 2], "Mono-Guided Synthesis", monoGuided_wav,
                        ("", xticks_mono, x_ticklabs_mono), speech_kwargs,
                        max_len=min_frame)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def draw_cond_syn_pitch(out_png="monGuideAttn_refSynMel.png"):
    """
    cond_pitch syn_pitch
    """
    show_spk, show_emo, speech_id = "spk0019", "Surprise", "spk0019_Surprise_ref3_syn1" #"spk0019_Surprise_ref4_syn0"

    # 1. Get conditioning pe type: ref_pe, pred_pe, fuse (2)  -> Get synthesized speech
    psdcond_path = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_fuse/psdcond_mdit_random.json"
    with open(psdcond_path, "r") as f:
        psdcond_dict = json.load(f)

    ref_txt_index = psdcond_dict[show_spk][show_emo]["ref_pe_m05_f02"]["speechid"].index(speech_id)
    con_ref_p = psdcond_dict[show_spk][show_emo]["ref_pe_m05_f02"]["pitch_cond"][ref_txt_index]  # condition 1
    con_fuse_p_02 = psdcond_dict[show_spk][show_emo]["ref_pred_add_m05_f02"]["pitch_cond"][ref_txt_index]
    con_fuse_p_04 = psdcond_dict[show_spk][show_emo]["ref_pred_add_m05_f04"]["pitch_cond"][ref_txt_index]
    con_pred_p = psdcond_dict[show_spk][show_emo]["none_m05_f02"]["pitch_cond"][ref_txt_index]

    # 2. Get syn pe:
    psd_path = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_fuse/psd_ablation.json"
    with open(psd_path, "r", encoding="utf-8") as f:
        psd_dict = json.load(f)
    ref_txt_index = psd_dict[show_spk][show_emo]["ref_pe_m05_f02"]["speechid"].index(speech_id)
    audio_ref_pe = psd_dict[show_spk][show_emo]["ref_pe_m05_f02"]["pitch"][ref_txt_index]
    audio_fuse_pe_02 = psd_dict[show_spk][show_emo]["ref_pred_add_m05_f02"]["pitch"][ref_txt_index]
    audio_fuse_pe_04 = psd_dict[show_spk][show_emo]["ref_pred_add_m05_f04"]["pitch"][ref_txt_index]
    audio_pred_pe = psd_dict[show_spk][show_emo]["none_m05_f02"]["pitch"][ref_txt_index]

    # 3 draw
    fig, axs = plt.subplots(1, 2, figsize=(24, 10))
    labels = ["reference pitch", "fused pitch ($\gamma=0.2$)", "fused pitch ($\gamma=0.4$)", "predicted pitch"]
    x_ltl_set = ("frame", "", "")
    y_ltl_set = ("HZ", "", "")

    plot_lines(axs[0], "Pitch for conditioning", [con_ref_p, con_fuse_p_02, con_fuse_p_04, con_pred_p], labels, x_ltl_set, y_ltl_set)
    plot_lines(axs[1], "Pitch of synthesized speech", [audio_ref_pe, audio_fuse_pe_02, audio_fuse_pe_04, audio_pred_pe], labels, x_ltl_set, y_ltl_set)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)


def synthesize_sample(ref_path, syn_txt, mono_guide_delta=(), fuse_beta=()):
    """
    Synthesize speech given style, text with different monoGuide strength and fuse_strength
    return:

    """
    # set model config
    model_root_dir = "/home/rosen/ckpt/styletts2_libriTTS/"
    model_name = "mdit_cfm_v10"
    second_model_path, second_config = model_root_dir + model_config[model_name][0], model_root_dir + \
                                       model_config[model_name][1]
    second_model, sampler, model_params = get_second_model(ckpt=second_model_path, config_f=second_config)

    # synthesize speech
    save_lines_path = ""

    ## Test on band_sigmas, save wavs, attn_maps
    inference_args_copy = inference_args.copy()
    for delta in mono_guide_delta:
        inference_args_copy["mono_guide_delta"] = delta
        audio, (ref_p, pred_p, fuse_p_02),  _, attn_maps, pred_dur = inference_second(syn_txt, ref_path,
                                                                                             second_model, sampler,
                                                                                             model_params,
                                                                                             **inference_args_copy)
        wav_f = os.path.basename(ref_path) + str(delta).replace(".", "") + ".wav"
        attn_map_f = os.path.basename(ref_path) + str(delta).replace(".", "") + "attn.npy"
        librosa.save(audio, wav_f)
        np.save(attn_maps, attn_map_f)


    ## Test on ref_pe, pred_pe and fuse with different fuse_beta, save wavs, cond/syn pe
    inference_args_copy = inference_args.copy()
    inference_args_copy["mix_ref_pe_type"] = "none"
    audio_pred_pe, (ref_p, pred_p, _),  _, pred_dur = inference_second(syn_txt, ref_path, second_model, sampler,
                                                                   model_params, **inference_args_copy)

    inference_args_copy["mix_ref_pe_type"] = "ref_pe"
    audio_ref_pe, (ref_p, pred_p, _), _, pred_dur = inference_second(syn_txt, ref_path, second_model, sampler,
                                                                  model_params, **inference_args_copy)
    inference_args_copy["mix_ref_pe_type"] = "ref_pred_add"
    inference_args_copy["fuse_beta"] = 0.2
    audio_fuse_pe_02, (ref_p, pred_p, fuse_p_02),  _, pred_dur = inference_second(syn_txt, ref_path, second_model,
                                                                              sampler, model_params,
                                                                              **inference_args_copy)
    inference_args["fuse_beta"] = 0.4
    audio_fuse_pe_04, (ref_p, pred_p, fuse_p_04),  _, pred_dur = inference_second(syn_txt, ref_path, second_model,
                                                                              sampler, model_params,
                                                                              **inference_args_copy)
    save_lines = {
        "condition_p": [ref_p, fuse_p_02, fuse_p_04, pred_p],
        "synthesize_p": [audio_ref_pe, audio_fuse_pe_02, audio_fuse_pe_04, audio_pred_pe]
    }
    with open(save_lines_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(save_lines, sort_keys=False, indent=4))

if __name__ == '__main__':
    img_dir = "/home/rosen/ckpt/exp/mdit_tts_esd/img_out"
    speech_id = "spk0019_Surprise_ref3_syn0"  # "spk0019_Surprise_ref4_syn0"
    band_sigmas = [0.2, 0.5, 0.8, -1]


    attn_mel_png = os.path.join(img_dir, "attn_mel_v5.png")
    cond_syn_pitch_png = os.path.join(img_dir, "cond_syn_pitch.png")

    band_attn_png = os.path.join(img_dir, "band_attn_b0_h3_v2.png")

    #draw_bandMatrix_monoGuideAttns(speech_id, band_sigmas, show_t=0, show_b=0, show_h=3, out_png=band_attn_png)
    draw_monGuideAttn_refSynMel2(speech_id="spk0019_Surprise_ref3_syn0", band_sigmas=("none_m02_f02", "none_mm10_f02"),
                                out_png=attn_mel_png, show_t=0, show_b=0, show_h=0)  # none_mm10_f02
    #draw_cond_syn_pitch(cond_syn_pitch_png)