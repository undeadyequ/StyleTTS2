import numpy as np
import matplotlib.pyplot as plt
import json

# Optional (for KDE). If you don't have SciPy, the code will still run (no KDE).
try:
    from scipy.stats import gaussian_kde
    USE_SCIPY_KDE = True
except Exception:
    USE_SCIPY_KDE = False


def _mean_pitch_exclude_unvoiced(pitch_seq):
    """Mean of voiced frames only (pitch > 0). Returns np.nan if no voiced frames."""
    p = np.asarray(pitch_seq, dtype=np.float32)
    voiced = p[p > 0.0]
    return float(voiced.mean()) if voiced.size else np.nan


def build_mean_pitch_dict(emo_model_pe_dict, prosody_type="pitch"):
    """
    Input:
      emo_model_pe_dict[emo][model]["pitch"] = list_of_utterances
      each utterance is a list/array of per-frame pitch with unvoiced=0

    Output:
      mean_pitch_dict[emo][model] = [utt_mean_pitch_1, utt_mean_pitch_2, ...]
    """
    mean_pitch_dict = {}
    for emo, model_dict in emo_model_pe_dict.items():
        mean_pitch_dict[emo] = {}
        for model, pe in model_dict.items():
            pitch_utts = pe.get(prosody_type, [])
            means = []
            for utt_pitch in pitch_utts:
                m = _mean_pitch_exclude_unvoiced(utt_pitch)
                if not np.isnan(m):
                    means.append(m)
            mean_pitch_dict[emo][model] = means
    return mean_pitch_dict


def plot_mean_f0_hist_kde_apsipa(
    emo_model_pe_dict,
    emotions,                # EXACTLY 5 emotions (list of 5 strings)
    models,                  # EXACTLY 6 models (list of 6 strings)
    bins=28,
    kde_grid_points=512,
    density=True,
    x_percentile_clip=(0.5, 99.5),   # robust xlim across all data
    savepath=None,
    prosody_type="pitch",
    middleValue={},
    figsize=(24, 12),
    display_names=None,      # dict mapping model key -> display name for legend
    xlabel=None,             # custom x-axis label (default: "Mean F0 (Hz)")
):
    """
    APSIPA-style figure:
      - 2x3 subplots (five emotions: 3 in row 1, 2 in row 2)
      - each subplot overlays 6 models: histogram + KDE of per-utterance mean F0
      - unvoiced frames (pitch==0) excluded before per-utterance mean
      - single shared legend below, clean axes, serif fonts
    """
    if len(emotions) != 5:
        raise ValueError(f"Need exactly 5 emotions for 5 sub-images, got {len(emotions)}.")

    # Default display names: use model key as-is if no mapping provided
    if display_names is None:
        display_names = {}
    _dn = lambda m: display_names.get(m, m)

    if len(middleValue) != 0:
        mean_pitch_dict = middleValue.copy()
    else:
        mean_pitch_dict = build_mean_pitch_dict(emo_model_pe_dict, prosody_type)

    # Collect all values for robust global x-limits
    all_vals = []
    for emo in emotions:
        for m in models:
            all_vals.extend(mean_pitch_dict.get(emo, {}).get(m, []))
    if len(all_vals) == 0:
        raise RuntimeError("No voiced mean-F0 values found. Check pitch arrays / pitch==0 masking.")
    all_vals = np.asarray(all_vals, dtype=np.float32)

    lo_p, hi_p = x_percentile_clip
    xmin = float(np.percentile(all_vals, lo_p))
    xmax = float(np.percentile(all_vals, hi_p))
    if xmax <= xmin:
        xmin, xmax = float(all_vals.min()), float(all_vals.max())
    pad = 0.06 * (xmax - xmin + 1e-6)
    xmin -= pad
    xmax += pad
    xgrid = np.linspace(xmin, xmax, kde_grid_points)

    # ---- APSIPA-ish styling (enlarged so fonts match caption after shrink) ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 36,
        "axes.titlesize": 36,
        "axes.labelsize": 36,
        "legend.fontsize": 36,
        "xtick.labelsize": 36,
        "ytick.labelsize": 36,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(
        2, 3,
        figsize=figsize,
        sharex=True,
        sharey=False,
        constrained_layout=False
    )
    axes_flat = axes.flatten()
    # Hide unused 6th subplot (row 1, col 2)
    axes_flat[5].set_visible(False)

    # Keep handles for a single shared legend
    legend_handles = {}
    # Slightly different alphas/linewidths for academic readability
    hist_alpha = 0.18
    kde_lw = 2.5

    for i, emo in enumerate(emotions):
        ax = axes_flat[i]
        ax.set_title(emo)

        # clean spines (typical academic style)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for model in models:
            vals = mean_pitch_dict.get(emo, {}).get(model, [])
            if len(vals) == 0:
                continue
            vals = np.asarray(vals, dtype=np.float32)

            # Histogram (density recommended when overlaying KDE)
            n, b, patches = ax.hist(
                vals,
                bins=bins,
                range=(xmin, xmax),
                density=density,
                alpha=hist_alpha,
                linewidth=0.0,
                label=_dn(model),
            )
            # KDE
            if USE_SCIPY_KDE and vals.size >= 2:
                kde = gaussian_kde(vals)
                y = kde(xgrid)
                (line,) = ax.plot(xgrid, y, linewidth=kde_lw)
                # store one handle per model for the shared legend
                if model not in legend_handles:
                    legend_handles[model] = line
            else:
                # if no KDE available, at least keep a histogram handle for legend
                if model not in legend_handles:
                    legend_handles[model] = patches[0] if len(patches) > 0 else None

        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xlim(xmin, xmax)

        if i % 3 == 0:
            ax.set_ylabel("Density" if density else "Count")
        ax.set_xlabel(xlabel or r"Mean $F_0$ (Hz)")
        ax.minorticks_on()

    # Shared legend below (6 columns for 6 models)
    handles = [legend_handles[m] for m in models if legend_handles.get(m) is not None]
    labels  = [_dn(m) for m in models if legend_handles.get(m) is not None]

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.2,
        handlelength=2.2,
    )

    # Tight layout while leaving room for legend
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.02)
    return fig, axes


def build_phoneme_std_dict(emo_model_std_dict, prosody_type="pitch_std"):
    """Pool all per-phoneme stds across utterances into a flat list per (emo, model).

    Input:
      emo_model_std_dict[emo][model]["pitch_std"] = [[ph1, ph2, ...], [ph1, ...], ...]
    Output:
      pooled[emo][model] = [all phoneme stds pooled]
    """
    pooled = {}
    for emo, model_dict in emo_model_std_dict.items():
        pooled[emo] = {}
        for model, data in model_dict.items():
            all_stds = []
            for utt_stds in data.get(prosody_type, []):
                all_stds.extend([s for s in utt_stds if not np.isnan(s)])
            pooled[emo][model] = all_stds
    return pooled


def plot_phoneme_std_hist_kde_apsipa(
    emo_model_std_dict, emotions, models,
    prosody_type="pitch_std", **kwargs,
):
    """Plot phoneme-level std histograms (reuses plot_mean_f0_hist_kde_apsipa)."""
    middle = build_phoneme_std_dict(emo_model_std_dict, prosody_type)
    if "pitch" in prosody_type:
        xlabel = r"Phoneme pitch std (Hz)"
    else:
        xlabel = r"Phoneme energy std"
    return plot_mean_f0_hist_kde_apsipa(
        None, emotions, models, middleValue=middle, xlabel=xlabel, **kwargs)


def extract_mean_f0_hist_kde_metadata(
    emo_model_pe_dict,
    emotions,
    models,
    bins=28,
    kde_grid_points=512,
    density=True,
    x_percentile_clip=(0.5, 99.5),
    exclude_models=None,
):
    """
    Returns a dict containing all metadata needed to reproduce
    histogram + KDE visualizations.
    """
    exclude_models = set(exclude_models or [])
    models = [m for m in models if m not in exclude_models]

    # ---- compute per-utterance mean F0 (voiced only) ----
    def mean_voiced(p):
        p = np.asarray(p)
        v = p[p > 0]
        return float(v.mean()) if v.size else np.nan

    mean_pitch = {}
    all_vals = []

    for emo in emotions:
        mean_pitch[emo] = {}
        for model in models:
            pitch_utts = emo_model_pe_dict.get(emo, {}).get(model, {}).get("pitch", [])
            vals = []
            for utt in pitch_utts:
                m = mean_voiced(utt)
                if not np.isnan(m):
                    vals.append(m)
            mean_pitch[emo][model] = np.asarray(vals, dtype=np.float32)
            all_vals.extend(vals)

    if len(all_vals) == 0:
        raise RuntimeError("No voiced mean-F0 values found.")

    all_vals = np.asarray(all_vals)
    xmin = np.percentile(all_vals, x_percentile_clip[0])
    xmax = np.percentile(all_vals, x_percentile_clip[1])
    pad = 0.06 * (xmax - xmin + 1e-6)
    xmin, xmax = xmin - pad, xmax + pad

    x_grid = np.linspace(xmin, xmax, kde_grid_points)

    # ---- extract metadata ----
    meta = {
        "global": {
            "xlim": [float(xmin), float(xmax)],
            "bins": bins,
            "density": density,
            "excluded_models": sorted(exclude_models),
        },
        "emotions": {},
    }

    for emo in emotions:
        meta["emotions"][emo] = {}
        for model in models:
            vals = mean_pitch[emo][model]
            if vals.size == 0:
                continue

            hist, bin_edges = np.histogram(
                vals,
                bins=bins,
                range=(xmin, xmax),
                density=density,
            )

            kde = gaussian_kde(vals)
            kde_y = kde(x_grid)

            meta["emotions"][emo][model] = {
                "num_samples": int(vals.size),
                "histogram": {
                    "bin_edges": bin_edges.tolist(),
                    "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
                    "values": hist.tolist(),
                    "bin_width": float(bin_edges[1] - bin_edges[0]),
                },
                "kde": {
                    "x": x_grid.tolist(),
                    "y": kde_y.tolist(),
                    "bandwidth": float(np.sqrt(kde.covariance)[0, 0]),
                },
                "raw_mean_f0": vals.tolist(),  # optional but useful
            }

    return meta



if __name__ == '__main__':
    # Example usage:
    emo_model_pe_dict = {
        "Neutral": {
            "monoDiT": {"pitch": [[0, 120, 121, 0, 123], [125, 0, 126, 127]]},
            "StyleTTS2": {"pitch": [[110, 112, 0, 113], [0, 115, 116]]},
        },
        "Angry": {
            "monoDiT": {"pitch": [[0, 120, 121, 0, 123], [125, 0, 126, 127]]},
            "StyleTTS2": {"pitch": [[110, 112, 0, 113], [0, 115, 116]]},
        },
        "Happy": {
            "monoDiT": {"pitch": [[0, 120, 121, 0, 123], [125, 0, 126, 127]]},
            "StyleTTS2": {"pitch": [[110, 112, 0, 113], [0, 115, 116]]},
        },
        "Sad": {
            "monoDiT": {"pitch": [[0, 120, 121, 0, 123], [125, 0, 126, 127]]},
            "StyleTTS2": {"pitch": [[110, 112, 0, 113], [0, 115, 116]]},
        },
        "Surprise": {
            "monoDiT": {"pitch": [[0, 120, 121, 0, 123], [125, 0, 126, 127]]},
            "StyleTTS2": {"pitch": [[110, 112, 0, 113], [0, 115, 116]]},
        },
    }
    # IN
    emo_model_pe_dict_f = "/home/rosen/ckpt/exp/mdit_tts_esd/psd_monoDiT_DiT_drawspeech_styletts2_hierspeech_reference.json"
    with open(emo_model_pe_dict_f, "r") as f:
        emo_model_pe_dict = json.load(f)
    emo_model_pe_dict = emo_model_pe_dict["spk0019"]

    emotions = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]   # 5 emotions
    models   = ["monoDiT", "styletts2", "hierspeech", "drawspeech", "reference"]  # 6 models
    fig, axes = plot_mean_f0_hist_kde_apsipa(emo_model_pe_dict, emotions, models, savepath="res/emo_model_pe_dict_pitch.png", prosody_type="pitch")

    # output meta
    meta = extract_mean_f0_hist_kde_metadata(
        emo_model_pe_dict,
        emotions=emotions,
        models=models,
    )
    with open("res/mean_f0_hist_kde_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

