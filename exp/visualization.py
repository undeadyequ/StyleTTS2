from cProfile import label
from typing import Any, Dict, Optional
from exp.exp_utils import convert_xydur_xybox, clean_phone
from exp.syllable import extend_phone2syl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import math

ordered_subs = ["Neutral", "Sad", "Angry", "Happy", "Surprise"]
ordered_lengend = ["reference", "ddpm_dit_cross", "ddpm_mdit_cross"]
STYLE_MAP = {
    # Emphasized (for immediate visual comparison)
    "reference": dict(
        color="black",
        linewidth=3.5,
        linestyle="--",
        marker="o",
        markersize=4,
        alpha=1.0,
        zorder=5,
    ),
    "monoDiT": dict(
        color="tab:blue",
        linewidth=3.0,
        linestyle="-",
        marker="o",
        markersize=4,
        alpha=1.0,
        zorder=4,
    ),
    # Baselines (lighter styling to reduce clutter)
    "styletts2": dict(
        color="tab:green",
        linewidth=1.6,
        linestyle="-",
        marker=None,
        alpha=0.6,
        zorder=2,
    ),
    "hierspeech": dict(
        color="tab:red",
        linewidth=1.6,
        linestyle="-",
        marker=None,
        alpha=0.6,
        zorder=2,
    ),
    "DiT": dict(
        color="tab:purple",
        linewidth=1.6,
        linestyle="-",
        marker=None,
        alpha=0.6,
        zorder=2
    ),
    "drawspeech": dict(
        color="tab:orange",
        linewidth=1.6,
        linestyle="-",
        marker=None,
        alpha=0.6,
        zorder=2
    ),
}

def vis_dual_utmos_rmse(dul_axis_data, out_pic):
    # Example data
    delta = dul_axis_data["x"]
    rmse = dul_axis_data["y1"]  # Example RMSE values (Hz)
    utmos = dul_axis_data["y2"]  # Example UTMOS values

    # Use academic-style fonts
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold"
    })

    fig, ax1 = plt.subplots(figsize=(4.5, 3.2), dpi=600)

    # Left y-axis (RMSE)
    ax1.plot(delta, rmse, marker="o", linestyle="-", color="black", label="RMSE")
    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel("RMSE (Hz)", color="black")
    ax1.set_xticks([0.2, 0.5, 0.8])
    ax1.set_xticklabels([r"$\delta=0.2$", r"$\delta=0.5$", r"$\delta=0.8$"])
    ax1.tick_params(axis="y", colors="black")
    ax1.set_ylim(0, 100)

    # Right y-axis (UTMOS)
    ax2 = ax1.twinx()
    ax2.plot(delta, utmos, marker="s", linestyle="--", color="grey", label="UTMOS")
    ax2.set_ylabel("UTMOS", color="black")
    ax2.tick_params(axis="y", colors="black")
    ax2.set_ylim(1, 5)

    # Grid (subtle, academic style)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Legend (combined for both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)

    # Save as high-quality academic figures
    plt.tight_layout()
    #plt.savefig("rmse_utmos_vs_delta.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_pic, bbox_inches="tight")

def vis_mono_guide_mask(images, out_png):
    """
    images = [
        "img_L100_d03.png", "img_L100_d05.png", "img_L100_d08.png",
        "img_L200_d03.png", "img_L200_d05.png", "img_L200_d08.png"
    ]
    """
    # INPUT
    row_labels = ["L=100", "L=200"]
    col_labels = [r"$\delta=0.3$", r"$\delta=0.5$", r"$\delta=0.8$"]

    # Academic style figure
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    # Adjust spacing: no vertical/horizontal gap
    plt.subplots_adjust(wspace=0.1, hspace=0.4)


    for i, ax in enumerate(axes.flat):
        img = mpimg.imread(images[i])
        ax.imshow(img)
        ax.axis("off")

    # Row labels (aligned on the left)
    for ax, row in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row, fontsize=14, rotation=90, labelpad=12, weight="bold")

    # Column labels (centered below)
    for ax, col in zip(axes[1], col_labels):
        ax.set_xlabel(col, fontsize=14, weight="bold")

    # Save in academic quality
    #plt.savefig("arranged_grid_academic.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")


def vis_matrix_attn(matrix, attns, out_png):
    """
    images = [
        "img_L100_d03.png", "img_L100_d05.png", "img_L100_d08.png",
        "img_L200_d03.png", "img_L200_d05.png", "img_L200_d08.png"
    ]
    """
    # INPUT
    row_labels = ["Mono-guidance \nmatrix", "Cross-attention \nmap"]
    col_labels = [r"$\delta=0.2$", r"$\delta=0.5$", r"$\delta=0.8$", r"$\delta=\varnothing$"]

    # Academic style figure
    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    # Adjust spacing: no vertical/horizontal gap
    #plt.subplots_adjust(wspace=0.1, hspace=0.4)

    for (i, j), ax in np.ndenumerate(axes):
        if i == 0:
            ax.imshow(matrix[j])
        else:
            ax.imshow(attns[j])
        ax.set_xticks([])
        ax.set_yticks([])

    # Row labels (aligned on the left)
    for ax, row in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row, fontsize=12, rotation=90, labelpad=12)

    # Column labels (centered below)
    for ax, col in zip(axes[0, :], col_labels):
        ax.set_title(col, fontsize=14, weight="bold")

    # Save in academic quality
    #plt.savefig("arranged_grid_academic.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")

def vis_matrix_attn2(matrix, attns, out_png):
    # ---------------------------
    # Global academic style
    # ---------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "text.usetex": False,              # Keep LaTeX-like math, but no full TeX compile
        "mathtext.fontset": "stix",        # Academic math font
        "figure.dpi": 300,
        "axes.linewidth": 0.6
    })

    # Row and column labels
    row_labels = [
        "Mono-guidance\nmatrix",
        "Cross-attention\nmap"
    ]
    col_labels = [
        r"$\delta = 0.2$",
        r"$\delta = 0.5$",
        r"$\delta = 0.8$",
        r"$\delta = \varnothing$"
    ]

    # Create figure (slightly wider to breathe)
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.6))
    fig.subplots_adjust(wspace=0.15, hspace=0.25)

    # Plot each subplot
    for (i, j), ax in np.ndenumerate(axes):
        img = matrix[j] if i == 0 else attns[j]
        ax.imshow(img, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])

        # Thicker frame for clarity
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # ---------------------------
    # Row labels (left side)
    # ---------------------------
    for ax, label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(label, fontsize=12, rotation=90, labelpad=12)

    # ---------------------------
    # Column labels (top)
    # ---------------------------
    for ax, label in zip(axes[0, :], col_labels):
        ax.set_title(label, fontsize=13, fontweight="bold", pad=10)

    # Save with publication quality
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

def vis_psd(pitch_dict_for_vis, energy_dict_for_vis, out_png, txt_id, show_ref_phone_on_line=True, ordered_lengend=("reference", "ddpm_dit_cross", "ddpm_mdit_cross")):
    rc_num = (3, 2)
    if len(ordered_lengend) == 3:
        legend_mark = ("*", "v", ".")  # reference, guide, ropePhone_guide
        legend_color = ("grey", "lightblue", "blue")
        legend_linestyle = ("dashed", "solid", "solid")
        xy_label_sub = ("frame", "Hz")
    else:
        legend_mark = ("*", "v", ".")  # reference, guide, ropePhone_guide
        legend_color = ("grey", "lightblue", "blue")
        legend_linestyle = ("dashed", "solid", "solid")
        xy_label_sub = ("frame", "Hz")

    title_extra = f"txt_ref{txt_id[0]}_txt{txt_id[1]}"
    print("3.1. do pitch comparation visualization refering to {}".format(" ".join(
        pitch_dict_for_vis["Angry"]["reference"][1])))

    out_pitch_png = out_png[:-4] + "_pitch.png"
    out_energy_png = out_png[:-4] + "_energy.png"

    ####### Pitch Contour ############
    compare_pitch_contour(
        pitch_dict_for_vis,
        rc_num,
        legend_mark=legend_mark,
        legend_color=legend_color,
        legend_linestyle=legend_linestyle,
        xy_label_sub=xy_label_sub,
        out_png=out_pitch_png,
        title="Pitch contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
        show_txt=show_ref_phone_on_line,
        ordered_lengend=ordered_lengend
    )
    print("5. do energy comparation visualization on {}st txt".format(txt_id))

    ####### Energy Contour ############
    compare_pitch_contour(
        energy_dict_for_vis,
        rc_num,
        legend_mark=legend_mark,
        legend_color=legend_color,
        legend_linestyle=legend_linestyle,
        xy_label_sub=xy_label_sub,
        out_png=out_energy_png,
        title="Energy contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
        show_txt=show_ref_phone_on_line,
        ordered_lengend=ordered_lengend
    )


def vis_psd_contour2(
    pitch_phonemes_dict,
    emotions=None,
    model_order=("reference", "monoDiT", "styleTTS2", "hierspeech+++"),
    suptitle=None,
    subtitle=None,
    ncols=2,
    figsize=(18, 10),
    xtick_stride=None,         # None = auto, or set e.g., 1 to show every phoneme
    rotate_xticks=0,
    y_label="Hz",
    x_label="Phoneme index",
    grid_alpha=0.2,
    legend_ncol=4,
    savepath=None,
    show=True,
):
    """
    Plot F0 (pitch) contours for multiple emotions and models.

    Input
    -----
    pitch_phonemes_dict : dict
        {emotion: {model_name: (pitch_list, phoneme_list), ...}}
        with len(pitch_list) == len(phoneme_list).

    Notes
    -----
    - The x-axis uses phoneme indices; tick labels are phoneme symbols.
    - If phoneme sequences differ across models for an emotion, each curve is plotted
      against its own index grid; tick labels are derived from the reference
      phoneme list when available.
    """
    model_rename_dict = {
        "monoDiT": "monoDiT-TTS",
        "styletts2": "StyleTTS2",
        "drawspeech": "Drawspeech",
        "DiT": "DiT-TTS",
        "hierspeech": "Hierspeech++",
        "reference": "Reference",
    }

    if emotions is None:
        emotions = list(pitch_phonemes_dict.keys())

    n = len(emotions)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14, y=0.98)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=11)

    legend_handles = None
    legend_labels = None

    for idx, emotion in enumerate(emotions):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.set_title(str(emotion), fontsize=12)

        emo_dict = pitch_phonemes_dict.get(emotion, {})
        if not emo_dict:
            ax.text(0.5, 0.5, f"No data for {emotion}", ha="center", va="center")
            ax.axis("off")
            continue

        # order the legend
        ordered_models = [m for m in model_order if m in emo_dict]
        remaining_models = [m for m in emo_dict.keys() if m not in ordered_models]
        models_to_plot = ordered_models + remaining_models

        for model_name in models_to_plot:
            pitch_list, phoneme_list = emo_dict[model_name]
            if len(pitch_list) != len(phoneme_list):
                raise ValueError(
                    f"Length mismatch for emotion={emotion}, model={model_name}: "
                    f"{len(pitch_list)} != {len(phoneme_list)}"
                )
            lengend_model_name = model_rename_dict[model_name]

            x = np.arange(len(pitch_list))
            style = STYLE_MAP.get(model_name, {})
            ax.plot(
                x,
                pitch_list,
                label=lengend_model_name,
                **style,
            )

        # Tick labels from monoDiT if available; otherwise fall back to the first model
        if "monoDiT" in emo_dict:
            base_pitch, base_phonemes = emo_dict["monoDiT"]
        #else:
        #    first_model = next(iter(emo_dict.keys()))
        #    _, base_phonemes = emo_dict[first_model]

        L = len(base_phonemes)
        if xtick_stride is None:
            stride = max(1, int(math.ceil(L / 24)))  # keep tick labels readable
        else:
            stride = max(1, int(xtick_stride))

        tick_pos = list(range(0, L, stride))
        tick_lab = [base_phonemes[i] for i in tick_pos]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=rotate_xticks)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=grid_alpha)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(legend_ncol, len(legend_labels)),
            frameon=True,
            bbox_to_anchor=(0.5, 0.915),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def vis_psd_enh(pitch_dict_for_vis, prosody_dict, out_png, txt_id, show_ref_phone_on_line=True, aux_range=None):
    rc_num = (3, 2)
    legend_mark = ("*", "v", ".")  # reference, guide, ropePhone_guide
    legend_color = ("grey", "lightblue", "blue")
    legend_linestyle = ("dashed", "solid", "solid")
    xy_label_sub = ("Frame", "Hz")

    #txt_n = len(prosody_dict["Angry"]["reference"]["phonemes"])   # should have ten
    real_text_id = max([i if txt_name[-1] == str(txt_id) else -1 for i, txt_name in
                        enumerate(prosody_dict["Angry"]["reference"]["speechid"])])  # psd["pitch"] is not sorted
    syn_phones = prosody_dict["Angry"]["reference"]["phonemes"][real_text_id]   # syn_phones should be same for all emotion

    title_extra = f"txt{txt_id}"
    print("3.1. do pitch comaration visualization on {}".format(" ".join(
        prosody_dict["Angry"]["reference"]["phonemes"][txt_id])))

    compare_pitch_contour(
        pitch_dict_for_vis,
        rc_num,
        legend_mark=legend_mark,
        legend_color=legend_color,
        legend_linestyle=legend_linestyle,
        xy_label_sub=xy_label_sub,
        xtickslab=syn_phones,
        out_png=out_png.format("enhance", title_extra.split(" ")[0]),
        title="Pitch contour of speech synthesized on reference, emoMix, proposed ({})".format(title_extra),
        show_txt=show_ref_phone_on_line,
        aux_range=aux_range,
    )

def vis_emo_crossAttn(attn_map_for_vis, out_png, show_txt=0, tick_gran="syllable"):
    """
    Show N pic for each model, each model include M subplot for each emotion
    Args:
        attn_map_for_vis:
        prosody_dict:
        out_png:
        show_txt:
        tick_gran:

    Returns:
    """
    # attn_map_for_vis -> {"model1": {"emo1": [np.array([syn_frames, ref_frames]), durations, phonemes] }}}
    for model_n, emo_attn_dict in attn_map_for_vis.items():  # {"model1": {"emo1": np.array([syn_frames, ref_frames])}}}
        rc_num = (3, 2)
        xy_label_sub = ("Reference frames", "Synthesis frames")
        title_extra = "non_parallel"
        show_attn_map(
            emo_attn_dict,
            rc_num,
            xlabel=xy_label_sub[0],
            ylabel=xy_label_sub[1],
            out_png=out_png[:-4] + "_" + model_n + ".png",
            # "result/{}_similarity_{}.png"
            title="Attention map in crossAttention after masking ({})".format(title_extra),
            tick_gran=tick_gran)

model_brev = {
    "mdit_librittsesd_cutdurspn_pe": "mdit_libritts"
}
## compare pitch contour
def compare_pitch_contour(
        sub_leg_data_dict,
        rc_num,
        legend_mark,
        legend_color,
        legend_linestyle,
        xy_label_sub,
        out_png,
        title="pitch contour",
        show_txt=True,
        aux_range=None,
        ordered_lengend=("reference", "ddpm_dit_cross", "ddpm_mdit_cross")
):
    """
    args:
    sub_leg_data_dict:
    {emotion:{modelA: ([1, 2, 5], [p1, p2]), modelB: ([2, 3, 4], [p1, ..])...}}
    rc_num: (2, 3)
    title: ""
    legend_mark: ["*", "o"]  # same len as len(subtitle.keys())  -> assert
    xy_label_sub: ("xlabel", "ylabel")
    Returns:
    """

    # fig size
    r_num, c_num = rc_num
    fig, axes = plt.subplots(r_num, c_num, figsize=(10 * (r_num / c_num), 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    fontsize = 12

    n = 0

    #for i, (sub, leg_data) in enumerate(sub_leg_data_dict.items()):
    for i, sub in enumerate(ordered_subs):
        r = int(n / c_num)
        c = int(n % c_num)
        axes[r, c].set_title(sub, fontsize=fontsize)

        #for j, (legend, data) in enumerate(leg_data.items()):
        for j, model in enumerate(ordered_lengend):
            pitch, phonemes = sub_leg_data_dict[sub][model]
            if not len(pitch) == len(phonemes):
                print(sub, model)
            x_value = range(len(pitch))
            label_name = model_brev[model] if model in model_brev else model
            axes[r, c].plot(x_value, pitch, label=label_name, marker=legend_mark[j], color=legend_color[j], linestyle=legend_linestyle[j])
            axes[r, c].set_xlabel(xy_label_sub[0])
            axes[r, c].set_ylabel(xy_label_sub[1])
            # axes[r, c].set_xlim(right=xticks_columns[psd])

            # show reference phoneme for non-parallel style
            if show_txt and model == "reference":
                for p_ind in range(len(phonemes)):
                    axes[r, c].text(x_value[p_ind] + 0.1, pitch[p_ind] + 0.2, phonemes[p_ind], ha='left', rotation=5, wrap=True, c="green", fontsize=fontsize)
            if model != "reference":
                axes[r, c].set_xticks(range(len(phonemes)))
                axes[r, c].set_xticklabels(phonemes, fontsize=fontsize)

        if r == 0 and c == 0:
          axes[r, c].legend(loc="upper left", ncol=3, fontsize=fontsize, bbox_to_anchor=(-0.1, 1.12, 1.2, 0.2))  # x, y,
        if aux_range is not None:
            for ax_x in aux_range:
                axes[r, c].axvline(x=ax_x, color="blue", linestyle="--", linewidth=0.3)
        n += 1
    fig.delaxes(axes[2, 1])
    fig.suptitle(title)
    fig.savefig(out_png, dpi=300)

def show_attn_map(
        sub_data_dict,
        rc_num,
        out_png,
        title="attn_map",
        tick_gran="phoneme",
        xlabel="Reference frames",
        ylabel="Synthesis frames",
        rotation=45,
        xy_label_font_size=12,
        need_auxline=False,
        need_rectangle=True,
        stress_xyLable_index=2,   # set to 100 if not use
        rect_line_width=0.5,
        rect_line_style="--"
):
    """
    Args:
    sub_data_dict:  {"emo1": [attn, syn_phone_durs, syn_phones, ref_phone_durs, ref_phones] }}}
    """
    r_num, c_num = rc_num
    fig, axes = plt.subplots(r_num, c_num, figsize=(10 * (r_num / c_num), 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    fontsize = 12
    n = 0

    for emo, contents in sub_data_dict.items():
        r = int(n / c_num)
        c = int(n % c_num)
        axes[r, c].set_title(emo, fontsize=fontsize)
        attn, syn_durs, syn_labels, ref_durs, ref_labels = contents
        pc = axes[r, c].pcolor(attn, cmap=plt.cm.Blues, alpha=0.9)

        #### 2. Set xy ticks, ticklabels, and labels
        syn_durs_inc, ref_durs_inc = [0], [0]
        syn_durs_inc.extend([sum(syn_durs[:i + 1]) for i in range(len(syn_durs))])
        ref_durs_inc.extend([sum(ref_durs[:i + 1]) for i in range(len(ref_durs))])
        syn_labels.append("")  # to align label to the left
        ref_labels.append("")  # to align label to the left
        #print("syn_durs_inc, syn_labels: {} {}".format(syn_durs_inc, syn_labels))
        #print("ref_durs_inc, ref_labels: {} {}".format(ref_durs_inc, ref_labels))
        if len(syn_durs_inc) != len(syn_labels) or len(ref_durs_inc) != len(ref_labels):
            raise IOError("durs and phones should have same lens {} {} {} {}".format(
                len(syn_durs_inc), len(syn_labels), len(ref_durs_inc), len(ref_labels)))

        axes[r, c].set_xticks(syn_durs_inc)
        axes[r, c].set_xticklabels(labels=syn_labels, fontsize=xy_label_font_size, rotation=rotation, ha="left")
        axes[r, c].set_yticks(ref_durs_inc)
        axes[r, c].set_yticklabels(labels=ref_labels, fontsize=xy_label_font_size, rotation=rotation, va="center")
        axes[r, c].set_xlabel(xlabel)
        axes[r, c].set_ylabel(ylabel)
        # set bold xyLabels
        for i, xlab in enumerate(axes[r, c].get_xticklabels()):
            if i == stress_xyLable_index:
                xlab.set_fontweight("bold")
        for i, ylab in enumerate(axes[r, c].get_yticklabels()):
            if i == stress_xyLable_index:
                ylab.set_fontweight("bold")

        #### 3. set auxiliary lines and rectangle
        if need_auxline:
            for ax_x in syn_durs_inc:
                axes[r, c].axvline(x=ax_x, color="blue", linestyle="--", linewidth=0.3)
            for ax_y in ref_durs_inc:
                axes[r, c].axhline(y=ax_y, color="blue", linestyle="--", linewidth=0.3)
        if need_rectangle:
            xywh_list = convert_xydur_xybox(syn_durs_inc, ref_durs_inc)  # get rectangle coordinate
            for i, (x, y, w, h) in enumerate(xywh_list):
                if i == stress_xyLable_index:
                    rect_line_width = 1.0
                    rect_line_style = "-"
                axes[r, c].add_patch(plt.Rectangle((x, y), w, h, ls=rect_line_style, ec="red", fc="none", linewidth=rect_line_width))
        fig.colorbar(pc, ax=axes[r, c])
        n += 1
    fig.suptitle(title)
    fig.delaxes(axes[2, 1])
    fig.savefig(out_png, dpi=300)


def show_two_attn_map(
        sub_data_dict,
        rc_num,
        out_png,
        title="attn_map",
        tick_gran="phoneme",
        xlabel="Reference frames",
        ylabel="Synthesis frames",
        rotation=45,
        xy_label_font_size=12,
        need_auxline=False,
        need_rectangle=True,
        stress_xyLable_index=2,   # set to 100 if not use
        rect_line_width=0.5,
        rect_line_style="--"
):
    """

    Args:
    sub_data_dict:  {"emo1": [attn, syn_phone_durs, syn_phones, ref_phone_durs, ref_phones] }}}
    """
    r_num, c_num = rc_num
    fig, axes = plt.subplots(r_num, c_num, figsize=(30, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    fontsize = 12
    n = 0

    for emo, contents in sub_data_dict.items():
        r = int(n / c_num)
        c = int(n % c_num)
        axes[c].set_title(emo, fontsize=fontsize)
        attn, syn_phone_durs, syn_phones, ref_phone_durs, ref_phones = contents
        pc = axes[c].pcolor(attn, cmap=plt.cm.Blues, alpha=0.9)

        #### 1. Change granularity of x, y ticks, labels
        if tick_gran == "syllable":
            syn_labels, syn_durs = extend_phone2syl(syn_phones, syn_phone_durs) # change phone to syllable
            ref_labels, ref_durs = extend_phone2syl(ref_phones, ref_phone_durs)
        else:
            syn_labels, syn_durs = clean_phone(syn_phones, syn_phone_durs)  # clean "", 11
            ref_labels, ref_durs = clean_phone(ref_phones, ref_phone_durs)

        #### 2. Set xy ticks, ticklabels, and labels
        syn_durs_inc, ref_durs_inc = [0], [0]
        syn_durs_inc.extend([sum(syn_durs[:i + 1]) for i in range(len(syn_durs))])
        ref_durs_inc.extend([sum(ref_durs[:i + 1]) for i in range(len(ref_durs))])
        syn_labels.append("")  # to align label to the left
        ref_labels.append("")  # to align label to the left
        #print("syn_durs_inc, syn_labels: {} {}".format(syn_durs_inc, syn_labels))
        #print("ref_durs_inc, ref_labels: {} {}".format(ref_durs_inc, ref_labels))
        if len(syn_durs_inc) != len(syn_labels) or len(ref_durs_inc) != len(ref_labels):
            raise IOError("durs and phones should have same lens {} {} {} {}".format(
                len(syn_durs_inc), len(syn_labels), len(ref_durs_inc), len(ref_labels)))
        axes[c].set_xticks(syn_durs_inc)
        axes[c].set_xticklabels(labels=syn_labels, fontsize=xy_label_font_size, rotation=rotation, ha="left")
        axes[c].set_yticks(ref_durs_inc)
        axes[c].set_yticklabels(labels=ref_labels, fontsize=xy_label_font_size, rotation=rotation, va="center")
        axes[c].set_xlabel(xlabel)
        axes[c].set_ylabel(ylabel)
        # set bold xyLabels
        for i, xlab in enumerate(axes[c].get_xticklabels()):
            if i == stress_xyLable_index:
                xlab.set_fontweight("bold")
        for i, ylab in enumerate(axes[c].get_yticklabels()):
            if i == stress_xyLable_index:
                ylab.set_fontweight("bold")

        #### 3. set auxiliary lines and rectangle
        if need_auxline:
            for ax_x in syn_durs_inc:
                axes[c].axvline(x=ax_x, color="blue", linestyle="--", linewidth=0.3)
            for ax_y in ref_durs_inc:
                axes[c].axhline(y=ax_y, color="blue", linestyle="--", linewidth=0.3)
        if need_rectangle:
            xywh_list = convert_xydur_xybox(syn_durs_inc, ref_durs_inc)  # get rectangle coordinate
            for i, (x, y, w, h) in enumerate(xywh_list):
                if i == stress_xyLable_index:
                    rect_line_width = 1.0
                    rect_line_style = "-"
                axes[c].add_patch(plt.Rectangle((x, y), w, h, ls=rect_line_style, ec="red", fc="none", linewidth=rect_line_width))
        fig.colorbar(pc, ax=axes[c])
        n += 1
    fig.suptitle(title)
    fig.savefig(out_png, dpi=300)

def _to_numpy(x):
    """Accept numpy or torch tensors."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

def vis_tbh_cross_attention(
    tbh_attn,
    time_labels=None,
    block_labels=None,
    head_labels=None,
    title=None,
    dashed_time_separator=True,
    savepath=None,
):
    """
    Plot cross-attention maps in a single figure with layout:
        rows = time_number * block_number   (2*2=4)
        cols = heads_num                    (4)

    Input:
        tbh_attn: array-like, shape (T=2, B=2, H=4, L, L)
                 attention weights or attention-like values.

    Layout (default row order):
        Row 0: time 0, block 0
        Row 1: time 0, block 1
        Row 2: time 1, block 0
        Row 3: time 1, block 1
    """
    A = _to_numpy(tbh_attn)
    assert A.ndim == 5, f"Expected 5D tensor (T,B,H,L,L), got shape {A.shape}"
    T, B, H, L1, L2 = A.shape
    assert (T, B, H) == (2, 2, 4), f"Expected (2,2,4, L, L), got {(T, B, H)}"
    assert L1 == L2, f"Expected square attention maps (L,L), got {(L1, L2)}"
    L = L1

    # -----------------------------
    # Academic-style figure settings
    # -----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
    })

    if time_labels is None:
        time_labels = [r"Time 1 (early)", r"Time 2 (late)"]
    if block_labels is None:
        block_labels = ["Block 1 (shallow)", "Block 2 (deep)"]
    if head_labels is None:
        head_labels = [f"Head {i+1}" for i in range(H)]

    # Row labels: (time, block) pairs
    row_pairs = [(t, b) for t in range(T) for b in range(B)]
    row_labels = [f"{block_labels[b]} · {time_labels[t]}" for (t, b) in row_pairs]

    # Create figure: 4 rows × 4 cols
    nrows = T * B
    ncols = H
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8.6, 7.8),
        constrained_layout=True,
    )

    # Shared color scale across all panels (recommended for fair comparison)
    vmin = float(np.min(A))
    vmax = float(np.max(A))

    images = []
    for r, (t, b) in enumerate(row_pairs):
        for h in range(H):
            ax = axes[r, h]
            im = ax.imshow(
                A[t, b, h],
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            images.append(im)

            # Column titles (heads)
            if r == 0:
                ax.set_title(head_labels[h])

            # Row labels on the first column only
            if h == 0:
                ax.set_ylabel(row_labels[r] + "\nAcoustic frame index")

            # Minimal ticks
            ax.set_xticks([0, L // 2, L - 1])
            ax.set_yticks([0, L // 2, L - 1])

            # X-axis label only on bottom row
            if r == nrows - 1:
                ax.set_xlabel("Prosody frame index")

    # One shared colorbar
    cbar = fig.colorbar(images[0], ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("Cross-attention weight")

    # Optional dashed separator between time groups:
    # between (time 0 rows) and (time 1 rows), i.e., between row 1 and row 2.
    if dashed_time_separator:
        # Use figure coordinates so it remains correct under layout changes.
        y_sep = axes[B - 1, 0].get_position().y0  # bottom of row 1 (0-index)
        sep = plt.Line2D(
            [0.06, 0.94],
            [y_sep, y_sep],
            transform=fig.transFigure,
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.8,
        )
        fig.add_artist(sep)

    # Optional title (often omitted in final paper; put in caption instead)
    if title is not None:
        fig.suptitle(title, y=1.02, fontsize=10)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()


def vis_tbh_cross_attention_time_grouped(
    tbh_attn,
    block_labels=None,
    time_labels=None,
    head_labels=None,
    dashed_time_separator=True,
    savepath=None,
    title=None,
):
    """
    Plot cross-attention maps with layout:
        rows = blocks (B=2)
        cols = times (T=2) grouped × heads (H=4) within each group
             = T * H = 8

    Column order:
        [time 0: head1 head2 head3 head4] | [time 1: head1 head2 head3 head4]
    Input:
        tbh_attn: shape (T=2, B=2, H=4, L, L)
    """
    A = _to_numpy(tbh_attn)
    assert A.ndim == 5, f"Expected (T,B,H,L,L), got {A.shape}"
    T, B, H, L1, L2 = A.shape
    assert (T, B, H) == (2, 2, 4), f"Expected (2,2,4,L,L), got {(T,B,H)}"
    if L1 != L2:
        tbh_attn = tbh_attn[:, :min(L1, L2), :min(L1, L2)]
        L1 = min(L1, L2)
        L2 = min(L1, L2)
    assert L1 == L2, f"Expected square maps (L,L), got {(L1,L2)}"
    L = L1

    # -----------------------------
    # Academic plotting style
    # -----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
    })

    if block_labels is None:
        block_labels = ["Block 1 (shallow)", "Block 6 (deep)"]
    if time_labels is None:
        time_labels = [r"Early diffusion ($t=t_1$)", r"Late diffusion ($t=t_T$)"]
    if head_labels is None:
        head_labels = [f"Head {i+1}" for i in range(H)]

    nrows = B
    ncols = T * H  # 8

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14.8, 4.8),
        constrained_layout=True,
    )

    # Shared scale for fair comparison
    vmin = float(A.min())
    vmax = float(A.max())

    images = []

    # Plot: columns grouped by time, with heads inside each group
    for b in range(B):
        for t in range(T):
            for h in range(H):
                col = t * H + h
                ax = axes[b, col]

                im = ax.imshow(
                    A[t, b, h],
                    origin="lower",
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                )
                images.append(im)

                # Head labels (top row only)
                if b == 0:
                    ax.set_title(head_labels[h])

                # Row labels (leftmost column only)
                if col == 0:
                    ax.set_ylabel(block_labels[b] + "\nAcoustic frame index")

                # Minimal ticks
                ax.set_xticks([0, L // 2, L - 1])
                ax.set_yticks([0, L // 2, L - 1])

                # X label only on bottom row
                if b == nrows - 1:
                    ax.set_xlabel("Prosodic frame index")

    # Shared colorbar
    cbar = fig.colorbar(images[0], ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("Cross-attention weight")

    # -----------------------------
    # Add time-group headers above each group of 4 heads
    # -----------------------------
    # Compute group centers in figure coordinates using axes positions.
    top_row_left = axes[0, 0].get_position()
    top_row_right = axes[0, ncols - 1].get_position()

    # A dynamic vertical offset proportional to axes height
    axes_h = top_row_left.height
    header_y = top_row_left.y1 + 0.35 * axes_h  # lift clearly above subplot titles

    for t in range(T):
        left_ax = axes[0, t * H]
        right_ax = axes[0, t * H + (H - 1)]

        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()

        x_center = 0.5 * (left_pos.x0 + right_pos.x1)

        fig.text(
            x_center,
            header_y,
            time_labels[t],
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=20,  # ensure on top
        )

    # Optional dashed vertical separator between the two time groups
    if dashed_time_separator:
        # x boundary between the two groups: right edge of col H-1 and left edge of col H
        left_group_right = axes[0, H - 1].get_position().x1
        right_group_left = axes[0, H].get_position().x0
        x_sep = 0.6 * (left_group_right + right_group_left)  # centered in the gap

        # y span exactly covering the grid of axes (all rows)
        grid_top = axes[0, 0].get_position().y1
        grid_bottom = axes[nrows - 1, 0].get_position().y0

        sep = plt.Line2D(
            [x_sep, x_sep],
            [grid_bottom, grid_top],
            transform=fig.transFigure,
            linestyle=(0, (4, 3)),  # clearer dash pattern than "--"
            linewidth=2.0,  # thicker
            color="black",  # higher contrast
            alpha=0.9,
            zorder=15,  # draw above axes
        )
        sep.set_clip_on(False)  # never clip by layout/axes
        fig.add_artist(sep)

    if title is not None:
        fig.suptitle(title, y=1.03, fontsize=11)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    #plt.show()