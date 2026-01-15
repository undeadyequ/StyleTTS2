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
import torch

from mpmath.libmp.libelefun import atan_newton

from exp.ref_aware_pe3 import fuse_prosody_smooth_additive
from utilities.vis import save_plot
from vis_data_adaptor import phone2syl, convert_vis_psd_json
import matplotlib.pyplot as plt

from visualization import vis_mono_guide_mask
from vis2 import plot_mel_with_pitch, plot_attn_with_rect, plot_lines, plot_mel_with_pitch2, plot_attn_with_rect2
#from inference_1_or_2 import inference_second, get_second_model
import numpy as np
from utilities.guide_mask import make_guided_attention_masks2
from visualization import vis_matrix_attn, vis_matrix_attn2, vis_psd_contour2, vis_tbh_cross_attention, vis_tbh_cross_attention_time_grouped
from itertools import accumulate
from exec_draw_two_pitch import plot_pitch_multi
from exec_histogram import plot_mean_f0_hist_kde_apsipa, extract_mean_f0_hist_kde_metadata, build_mean_pitch_dict
from exp_utils import replace_certain_key_value


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


def create_bandMatrix_monoGuideAttns(speech_id, band_sigmas, show_t=0, show_b=0, show_h=0, mono_ablation_dir=""):
    #mono_ablation_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono/monoDiT_ablation"
    mono_sigma_attn_dirs = ["none_m02_f02_attn", "none_m05_f02_attn", "none_m08_f02_attn", "none_mm10_f02_attn"]
    bandMatrixs = []
    monoGuideAttns = []
    for attn_dir in mono_sigma_attn_dirs:
        attn_path = os.path.join(mono_ablation_dir, "monoDiT_ablation", attn_dir, speech_id + ".npy")
        attn = read_attn(attn_path, show_t, show_b, show_h)
        monoGuideAttns.append(attn)
    for sigma in band_sigmas:
        ilens, olens = [monoGuideAttns[0].shape[-2]], [monoGuideAttns[0].shape[-1]]
        guide_matrix = 1 - make_guided_attention_masks2(ilens, olens, base_sigma=sigma, eps=0.002).cpu().numpy()
        if guide_matrix.mean() == 1:   # set all-one matrix to yellow, instead of blue
            guide_matrix[0, 0, 0] = 0
        bandMatrixs.append(guide_matrix.squeeze())
    return bandMatrixs, monoGuideAttns

def draw_bandMatrix_monoGuideAttns(speech_id, band_sigmas, show_t=0, show_b=0, show_h=0, out_png="band_matrix_mono_attn.png", mono_ablation_dir=""):
    """ draw bands and attns when generated speech_id, conditioned on differentband_sigmas
        band1 band2 band3
        attn1 attn2 attn3
    """
    bandMatrixs, monoGuideAttns = create_bandMatrix_monoGuideAttns(speech_id, band_sigmas, show_t=show_t, show_b=show_b, show_h=show_h, mono_ablation_dir=mono_ablation_dir)
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
        show_t=0, show_b=0, show_h=0,
        ablation_dir="/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v3"):

    set_academic_style()  # <<< NEW

    # Directories
    mono_ablation_dir = os.path.join(ablation_dir, "monoDiT_ablation")
    ref_dir = os.path.join(ablation_dir, "reference/random")
    attn_path = os.path.join(ablation_dir, "attn_mdit_random.json")

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

    plot_mel_with_pitch2(axs[1, 1], "W/O Mono-Guided", origin_wav,
                        ("", xticks, x_ticklabs), speech_kwargs, show_pitch=True,
                        max_len=min_frame)

    plot_mel_with_pitch2(axs[1, 2], "W/ Mono-Guided", monoGuided_wav,
                        ("", xticks_mono, x_ticklabs_mono), speech_kwargs, show_pitch=True,
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

def draw_pitch_energy_contours(psd_json_path, vis_psd_config, out_pitch_img, out_energy_img, vis_pitch_cmp_path, vis_energy_cmp_path, use_vis_data=False):
    # hyper
    show_spk, show_text = vis_psd_config["esd"]["show_spk"], vis_psd_config["esd"]["show_txt"]

    with open(psd_json_path, "r") as f:
        psd_cmp_dict = json.load(f)

    if use_vis_data:
        with open(vis_pitch_cmp_path, "r") as f:
            vis_pitch_phonemes_dict = json.load(f)
        with open(vis_energy_cmp_path, "r") as f:
            vis_energy_phonemes_dict = json.load(f)
    else:
        vis_pitch_phonemes_dict, vis_energy_phonemes_dict = convert_vis_psd_json(
            psd_cmp_dict["spk" + show_spk], show_text, cutpad_reference=False, save_dict=(vis_pitch_cmp_path, vis_energy_cmp_path))  # for_vis: {"emo1": {"model1": list(psd_len)}}}

    emotions = ["Neutral", "Sad", "Angry", "Happy", "Surprise"]

    model_orders = ("reference", "DiT", "styleTTS2", "monoDiT", "hierspeech", "drawspeech")
    title = "Pitch contours of reference and synthesized speech generated by monoDiT-TTS, DrawSpeech, StyleTTS2, DiT-TTS, and HierSpeech++."
    vis_psd_contour2(
        vis_pitch_phonemes_dict,
        emotions=emotions,
        model_order=model_orders,
        suptitle=None,
        xtick_stride=1,  # show every phoneme label (as in your screenshot)
        rotate_xticks=0,
        savepath=out_pitch_img,
    )
    vis_psd_contour2(
        vis_energy_phonemes_dict,
        emotions=emotions,
        model_order=model_orders,
        subtitle=None,
        xtick_stride=1,  # show every phoneme label (as in your screenshot)
        rotate_xticks=0,
        savepath=out_energy_img,
    )


def draw_tbh_cross_attn(tbh_attn, out_pitch_img="tbh_cross_attn.pdf", b=[0, 5]):
    # IN
    """
    L = 80
    tbh_attn = np.random.rand(2, 2, 4, L, L).astype(np.float32)
    tbh_attn = tbh_attn / (tbh_attn.sum(axis=-1, keepdims=True) + 1e-8)
    """
    times = torch.tensor([0, 1])
    blks = torch.tensor(b)
    heads = torch.tensor([0, 1, 2, 3])

    tt, bb, hh = torch.meshgrid(times, blks, heads, indexing="ij")
    tbh_attn_sel = tbh_attn[tt, bb, hh, ...]
    #tbh_attn_sel = tbh_attn[bb, 0, hh, ...]

    # OUT
    vis_tbh_cross_attention_time_grouped(
        tbh_attn_sel,
        time_labels=[r"$t=0$ (early)", r"$t=T$ (late)"],
        block_labels=["Block 1", "Block 6"],
        head_labels=["Head 1", "Head 2", "Head 3", "Head 4"],
        title=None,
        dashed_time_separator=False,
        savepath=out_pitch_img,
    )



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
    ablation_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v4"
    img_dir = "/home/rosen/ckpt/exp/mdit_tts_esd/img_out"

    # F7 AttnTBH_TrainMono;  F8 AttnTBH_noTrainMono (No inference monotonic)
    #attn_path = os.path.join(ablation_dir, "monoDiT_ablation", "none_mm10_f02_attn", speech_id + ".npy")
    PSDCONTOUR = False # Fig 3
    ATTNSIGMA = False # Fig 4
    ATTNMEL = False # Fig 5
    ATTNTBH = True  # Fig 6, 7
    PSDCONTOUR_SIGMA = False
    HISTORGRAM = False
    if PSDCONTOUR:
        root_dir = "/home/rosen/ckpt/exp/mdit_tts_esd"
        # IN
        #cmp_modelnames_combine = "monoDiT_DiT_drawspeech_styletts2_hierspeech_reference"    # monoDiT with fuse_add version
        cmp_modelnames_combine = "monoDiT_DiT_drawspeech_styletts2_hierspeech_reference_v2" # monoDiT with nono_05 version (ablation best)
        psd_json_path = os.path.join(root_dir, f"psdave_{cmp_modelnames_combine}.json")

        #show_text_esd = (4, 2)  # cmp (4, 2) or (4, 0)

        for ref_id in range(1):
            for syn_id in range(1):
                show_text_esd = (4, 2)  # cmp (4, 2) or (4, 0)
                vis_psd_config = {
                    "esd": {"show_spk": "0019", "show_emo": "Surprise", "tick_gran": "syllable", "show_txt": show_text_esd},
                    "libritts": {"show_spk": "0019", "show_emo": "Surprise", "tick_gran": "syllable", "show_txt": (3, 4)}}  # ref3:  15 - 19

                # OUT
                out_pitch_img = os.path.join(root_dir, f"img_out/psdcontour/contour_pitch_ref{show_text_esd[0]}_syn{show_text_esd[1]}.pdf")
                out_energy_img = os.path.join(root_dir, f"img_out/psdcontour/contour_energy_ref{show_text_esd[0]}_syn{show_text_esd[1]}.pdf")

                vis_pitch_path, vis_energy_path = (
                    os.path.join(root_dir, f"img_out/psdcontour/vis_pitch_{cmp_modelnames_combine}.json"),
                    os.path.join(root_dir, f"img_out/psdcontour/vis_energy_{cmp_modelnames_combine}.json"))   # OUTPUT VIS data
                vis_pitch_cmp_path, vis_energy_cmp_path = (os.path.join(root_dir, f"img_out/psdcontour/vis_pitch_{cmp_modelnames_combine}_cmp.json"),
                                                   os.path.join(root_dir, f"img_out/psdcontour/vis_energy_{cmp_modelnames_combine}_cmp.json"))    # VIS CMP data
                draw_pitch_energy_contours(psd_json_path, vis_psd_config, out_pitch_img, out_energy_img, vis_pitch_cmp_path, vis_energy_cmp_path, use_vis_data=True)


    if ATTNSIGMA:
        band_sigmas = [0.2, 0.5, 0.8, -1]
        band_attn_png = os.path.join(img_dir, "band_attn_b0_h3_v3.pdf")
        speech_id = "spk0019_Surprise_ref3_syn0"  # "spk0019_Surprise_ref4_syn0";     cmp: spk0019_Surprise_ref3_syn0
        draw_bandMatrix_monoGuideAttns(speech_id, band_sigmas, show_t=0, show_b=0, show_h=3, out_png=band_attn_png,
                                       mono_ablation_dir=ablation_dir)  # cmp: show_t=0, show_b=0, show_h=3

    if ATTNMEL:
        #attn_mel_png = os.path.join(img_dir, "attn_mel_v7.png")
        for ref_id in range(1):
            for syn_id in range(1):
                ref_id, syn_id = 0, 1
                attn_mel_png = os.path.join(img_dir, f"attn_mel_ref{ref_id}_syn{syn_id}.png")
                draw_monGuideAttn_refSynMel2(speech_id=f"spk0019_Surprise_ref{ref_id}_syn{syn_id}",
                                             band_sigmas=("none_m02_f02", "none_mm10_f02"),
                                             out_png=attn_mel_png, show_t=0, show_b=0, show_h=0,
                                             ablation_dir=ablation_dir)  # none_mm10_f02

    if ATTNTBH:
        attn_dir_notrainMono = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v4/monoDiT_ablation/none_mm10_f02_attn"
        attn_dir_notrainMono_infer05 = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v4/monoDiT_ablation/none_m05_f02_attn"
        attn_dir_trainMono = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v6/monoDiT_ablation/none_mm10_f02_attn"  # not used

        test_attn_dir = "/home/rosen/Project/StyleTTS2/res/monoStyle_compare2/mdit_cfm_v10_epoch48_esd_seed0_ref_pe_m10_fuse02_0.3_0.7_v10_epoch48_attn"
        test_attn_dir = "/home/rosen/Project/StyleTTS2/res/hierstyle_test/mdit_cfm_v10_epoch48_esd_03_07_attn"
        root_dir = "/home/rosen/ckpt/exp/mdit_tts_esd"

        for ref_ind in range(1):
            for syn_ind in range(1):
                ref_ind, syn_ind = 0, 3   # CHAMPION (0, 3)  (0, 0)
                for figName, attn_dir in zip(["test"], [test_attn_dir]):
                #for figName, attn_dir in zip(["trainMono", "noTrainMono", "noTrainMonoInfer05"], [attn_dir_trainMono, attn_dir_notrainMono, attn_dir_notrainMono_infer05]):
                    #attn_path = os.path.join(attn_dir, f"spk0019_Surprise_ref{ref_ind}_syn{syn_ind}.npy")
                    attn_path = os.path.join(attn_dir, "spk0019_Angry_ref2_syn1.npy")
                    attn_maps = np.load(attn_path, allow_pickle=True)
                    out_pitch_img = os.path.join(root_dir, "img_out/tbh_cross_attn", f"tbh_cross_attn_ref{ref_ind}_syn{syn_ind}_{figName}_v17.pdf")
                    out_pitch_img = "/home/rosen/Project/StyleTTS2/res/hierstyle_test/tbh_hier_style_v10.png"
                    draw_tbh_cross_attn(attn_maps, out_pitch_img, b=[3, 4])

    if PSDCONTOUR_SIGMA:
        root_dir = "/home/rosen/ckpt/exp/mdit_tts_esd"
        ablation_dir = "/home/rosen/ckpt/exp/mdit_tts_esd_ablation_mono_v4"
        ref_id, syn_id = 0, 1

        # cmp: ref0_syn1
        for ref_id in range(5):
            for syn_id in range(5):
                wav_paths = [
                    os.path.join(ablation_dir, "monoDiT_ablation/none_m02_f02",
                                 f"spk0019_Angry_ref{ref_id}_syn{syn_id}.wav"),
                    os.path.join(ablation_dir, "monoDiT_ablation/none_m05_f02",
                                 f"spk0019_Angry_ref{ref_id}_syn{syn_id}.wav"),
                    os.path.join(ablation_dir, "monoDiT_ablation/none_m08_f02",
                                 f"spk0019_Angry_ref{ref_id}_syn{syn_id}.wav"),
                    os.path.join(ablation_dir, "monoDiT_ablation/none_mm10_f02",
                                 f"spk0019_Angry_ref{ref_id}_syn{syn_id}.wav"),
                    os.path.join(ablation_dir, "reference/random", f"spk0019_Angry_ref{ref_id}.wav"),
                ]
                labels = ["m02", "m05", "m08", "mm10", "reference"]

                out_pitch_img = os.path.join(root_dir, "img_out/psdcontour_sigma",
                                             f"pitch_contour_sigma_{ref_id}_syn{syn_id}.png")
                plot_pitch_multi(
                    wav_paths, labels,
                    sr=24000, hop_length=300,
                    f0_floor=50, f0_ceil=600,
                    normalize_time=True,  # set False if they’re same length
                    out_pdf=out_pitch_img
                )
    """
    t, b, h = 0, 0, 3
    for b in range(4):
        for h in range(4):
            attn = read_attn(attn_path, t, b, h)  # 0:vert 1:mono  2:pos  3:sick_vert
            save_plot(attn, f"attn_b{b}_h{h}.png")
    """
    #cond_syn_pitch_png = os.path.join(img_dir, "cond_syn_pitch.png")
    #draw_cond_syn_pitch(cond_syn_pitch_png)  # not used currently

    if HISTORGRAM:
        # CONFIG
        root_dir = "/home/rosen/ckpt/exp/mdit_tts_esd"
        emotions = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]  # 5 emotions
        models = ["monoDiT", "styletts2", "hierspeech", "drawspeech", "reference"]  # 6 models

        # IN
        emo_model_pe_dict_f = "/home/rosen/ckpt/exp/mdit_tts_esd/psd_monoDiT_DiT_drawspeech_styletts2_hierspeech_reference.json"
        replacement_dict_path = "/home/rosen/ckpt/exp/mdit_tts_esd_multiversion/psd_monoDiT_reference_0307_seed.json"

        # OUT
        out_img = os.path.join(root_dir, "img_out/histogram", f"histogram_diffseeds_cmp.pdf")


        # replace monoDiT content or not in original dict
        if False:
            with open(emo_model_pe_dict_f, "r") as f:
                emo_model_pe_dict = json.load(f)
            emo_model_pe_dict = emo_model_pe_dict["spk0019"]
        else:
            emo_model_pe_dict = replace_certain_key_value(emo_model_pe_dict_f, replacement_dict_path, replaced_dict_path="",
                                                          key_depth=2, key_name="monoDiT")["spk0019"]
        # create middle dict
        if False:
            mean_pitch_dict = build_mean_pitch_dict(emo_model_pe_dict, "pitch")
            with open(out_img.replace(".png", ".json"), "w") as f:
                json.dump(mean_pitch_dict, f, indent=2)
        # read middle dict
        middel_dict_path = "/home/rosen/ckpt/exp/mdit_tts_esd/img_out/histogram/histogram_diffseeds_cmp.json"
        with open(middel_dict_path, "r") as f:
            mean_pitch_dict = json.load(f)
        fig, axes = plot_mean_f0_hist_kde_apsipa(emo_model_pe_dict, emotions, models, savepath=out_img, prosody_type="pitch",
                                                 middleValue=mean_pitch_dict)
        """
        
        meta = extract_mean_f0_hist_kde_metadata(
            emo_model_pe_dict,
            emotions=emotions,
            models=models)
        with open("res/mean_f0_hist_kde_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        """