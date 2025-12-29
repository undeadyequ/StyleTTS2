"""
plot_attn_with_rectangle

"""
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import random
import matplotlib.patches as patches
import whisper
import pyworld as pw
import torch
from sympy.printing.pretty.pretty_symbology import line_width


def plot_attn_with_rect(ax, title, attn, x_ltl_set, y_ltl_set, kwargs):
    rect_line_width = 0.5
    rect_line_style = "--"
    xlabel, xticks, x_ticklabs = x_ltl_set
    ylabel, yticks, y_ticklabs = y_ltl_set

    ax.set_title(title, fontsize=kwargs["fontsize"])
    #pc = ax.pcolor(attn, cmap=plt.cm.Blues, alpha=0.9)
    ax.imshow(attn.T, cmap="viridis", aspect="auto", origin="lower")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlabel, fontsize=kwargs["fontsize"])
    ax.set_ylabel(ylabel, fontsize=kwargs["fontsize"], rotation=kwargs["yticklabel_rotation"], loc="top")

    if x_ticklabs is not None:
        ax.set_xticklabels(labels=x_ticklabs, fontsize=kwargs["fontsize"], rotation=kwargs["x_rotation"], ha="left")
    else:
        ax.set_xticklabels([], visible=False)
    if y_ticklabs is not None:
        ax.set_yticklabels(labels=y_ticklabs, fontsize=kwargs["fontsize"], rotation=kwargs["y_rotation"], va="bottom")
    else:
        ax.set_yticklabels([], visible=False)

    #### bold label
    if kwargs["xy_ticklabs_bold_index"][0] is not None:
        x_bindexs, y_bindexs = kwargs["xy_ticklabs_bold_index"]
        for i, xlab in enumerate(ax.get_xticklabels()):
            if i in x_bindexs:
                xlab.set_fontweight("bold")
        for i, ylab in enumerate(ax.get_yticklabels()):
            if i == y_bindexs:
                ylab.set_fontweight("bold")

    #### 3. draw auxiliary lines for xyticks
    if kwargs["xy_auxline"][0] is not None:
        for ax_x in xticks:
            ax.axvline(x=ax_x, color="blue", linestyle="--", linewidth=0.3)
        for ax_y in xticks:
            ax.axhline(y=ax_y, color="blue", linestyle="--", linewidth=0.3)

    ### 4. draw rectangle for xyticks
    if kwargs["xy_rectangle"] is not None:
        for i, (x, y, w, h) in enumerate(kwargs["xy_rectangle"]):
            if i in x_bindexs:
                rect_line_width = 1.0
                rect_line_style = "-"
            ax.add_patch(plt.Rectangle((x, y), w, h, ls=rect_line_style, ec="red", fc="none", linewidth=rect_line_width))
    #ax.legend(loc='upper right')

def plot_attn_with_rect2(ax, title, attn, x_ltl_set, y_ltl_set, kwargs):

    xlabel, xticks, x_ticklabs = x_ltl_set
    ylabel, yticks, y_ticklabs = y_ltl_set

    ax.set_title(title, pad=8)

    ax.imshow(attn.T, cmap="viridis", aspect="auto", origin="lower")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Tick labels
    if x_ticklabs is not None:
        ax.set_xticklabels(x_ticklabs, fontsize=kwargs["fontsize"],
                           rotation=kwargs["x_rotation"], ha="left")
    else:
        ax.set_xticklabels([])

    if y_ticklabs is not None:
        ax.set_yticklabels(y_ticklabs, fontsize=kwargs["fontsize"],
                           rotation=kwargs["y_rotation"], va="bottom")
    else:
        ax.set_yticklabels([])

    # Bold specific ticks
    x_bold, y_bold = kwargs["xy_ticklabs_bold_index"]
    for i, xt in enumerate(ax.get_xticklabels()):
        if i in x_bold:
            xt.set_fontweight("bold")
    for i, yt in enumerate(ax.get_yticklabels()):
        if i in y_bold:
            yt.set_fontweight("bold")

def plot_mel_with_pitch(ax, title, speech, x_ltl_set, kwargs, show_pitch=False, max_len=-1):
    y, sr = librosa.load(speech, sr=None)
    #hop_length, sr, n_fft, n_mels = 200, 16000, 1024, 128
    hop_length, sr, n_fft, n_mels = 300, 24000, 2048, 128  # 128 for clear
    xlabel, xticks, x_ticklabs = x_ltl_set

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000, win_length=1200)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = S_dB[...,:max_len]
    librosa.display.specshow(S_dB, x_axis=None, y_axis='mel', sr=sr, hop_length=hop_length, cmap='magma', ax=ax, fmax=8000, win_length=1200)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
        fmax=8000,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Estimate pitch (f0)
    if show_pitch:
        pitch, t = pw.dio(y.astype(np.float64), sr, frame_period=hop_length / sr * 1000)
        pitch = pw.stonemask(y.astype(np.float64), pitch, t, sr)
        times = librosa.times_like(pitch, sr=sr, hop_length=hop_length)
        ax.plot(times, pitch, color='cyan', linewidth=1.5, label='Pitch')
        print("speech:{} len of f0 {} and x_ticks {}".format(speech, len(pitch), xticks[-1]))
    ax.set_title(title)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels=x_ticklabs, fontsize=kwargs["fontsize"], rotation=kwargs["x_rotation"], ha="left")
    #ax.set_xticks(xticks)
    #ax.set_xlabel(xlabel, fontsize=kwargs["fontsize"])
    #if x_ticklabs is not None:
    #    ax.set_xticklabels(labels=x_ticklabs, fontsize=kwargs["fontsize"], rotation=kwargs["x_rotation"], ha="left")
    #ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)

    #### bold label
    if kwargs["xy_ticklabs_bold_index"][0] is not None:
        x_bindexs, y_bindexs = kwargs["xy_ticklabs_bold_index"]
        for i, xlab in enumerate(ax.get_xticklabels()):
            if i in x_bindexs:
                xlab.set_fontweight("bold")
        for i, ylab in enumerate(ax.get_yticklabels()):
            if i == y_bindexs:
                ylab.set_fontweight("bold")

    ### 4. draw rectangle for xyticks
    if kwargs["xy_rectangle"] is not None:
        for i, (x, y, w, h) in enumerate(kwargs["xy_rectangle"]):
            if i in x_bindexs:
                rect_line_width = 1.0
                rect_line_style = "-"
            ax.add_patch(plt.Rectangle((x, y), w, h, ls=rect_line_style, ec="red", fc="none", linewidth=rect_line_width))
    #ax.legend(loc='upper right')

    # Random rectangle dimensions in time-mel space
    """
    duration = times[-1]
    mel_bins = S.shape[0]
    x_start = random.uniform(0, duration * 0.8)
    width = random.uniform(0.5, duration - x_start)
    y_start = random.uniform(0, mel_bins * 0.8)
    height = random.uniform(5, mel_bins - y_start)
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)
    y_bottom_hz = mel_freqs[int(y_start)]
    y_top_hz = mel_freqs[int(min(y_start + height, mel_bins - 1))]
    rect = patches.Rectangle(
        (x_start, y_bottom_hz), width, y_top_hz - y_bottom_hz,
        linewidth=2, edgecolor='lime', facecolor='none', label='Random Rectangle'
    )
    ax.add_patch(rect)
    """


def plot_mel_with_pitch2(
    ax, title, speech, x_ltl_set, kwargs,
    show_pitch=False, max_len=-1,
    f0_min=50.0, f0_max=600.0   # 🔑 fixed pitch range
):
    y, sr = librosa.load(speech, sr=None)
    hop_length, n_fft, n_mels = 300, 2048, 128
    win_length, fmax = 1200, 8000

    ylabel, xticks, x_ticklabs = x_ltl_set

    # --- Mel-spectrogram ---
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels, fmax=fmax,
        power=2.0
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    if max_len > 0:
        S_dB = S_dB[:, :max_len]

    librosa.display.specshow(
        S_dB,
        y_axis="mel",
        fmax=fmax,
        cmap="magma",
        ax=ax
    )

    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel if ylabel is not None else "")

    # --- Fix x-limits to visible mel frames ---
    n_frames = S_dB.shape[1]
    ax.set_xlim(0, n_frames - 1)

    # --- Custom x ticks (frame-based) ---
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        x_ticklabs,
        fontsize=kwargs["fontsize"],
        rotation=kwargs["x_rotation"],
        ha="left"
    )
    ax.set_xlabel("")

    # Bold tick labels
    if kwargs.get("xy_ticklabs_bold_index", (None, None))[0] is not None:
        x_bold, _ = kwargs["xy_ticklabs_bold_index"]
        for i, lab in enumerate(ax.get_xticklabels()):
            if i in x_bold:
                lab.set_fontweight("bold")

    # --- Pitch contour (fixed y-range across figures) ---
    if show_pitch:
        frame_period_ms = hop_length / sr * 1000.0
        f0, t = pw.dio(y.astype(np.float64), sr, frame_period=frame_period_ms)
        f0 = pw.stonemask(y.astype(np.float64), f0, t, sr)

        if max_len > 0:
            f0 = f0[:max_len]

        x_frames = np.arange(len(f0))

        ax_f0 = ax.twinx()
        ax_f0.plot(x_frames, f0, lw=1.2, color="cyan")
        ax_f0.set_xlim(0, n_frames - 1)

        # 🔑 FIXED pitch scale for cross-figure comparison
        ax_f0.set_ylim(f0_min, f0_max)
        ax_f0.set_ylabel("F0 (Hz)")
        ax_f0.tick_params(axis="y", labelsize=kwargs["fontsize"])
        ax_f0.spines["right"].set_alpha(0.6)


def plot_lines(ax, title, lines, labels, x_ltl_set, y_ltl_set):
    colors = ["blue", "red", "green", "purple"]
    linestyles = ["-", "--", ":", "-."]
    alphas = [1.0, 1.0, 1.0, 1.0]
    linewidths = [2.0, 2.0, 2.0, 2.0]

    xlabel, _, _ = x_ltl_set
    ylabel, _, _ = y_ltl_set

    # axis labels and ticks
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    #ax.set_xticks(xticks)
    #ax.set_xticklabels(x_ticklabs)
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(y_ticklabs)

    T = len(lines[0])
    for i, (line, label) in enumerate(zip(lines, labels)):
        if line != T:
            line = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(line)), line)
        ax.plot(
            line,
            label=label,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            alpha=alphas[i % len(alphas)],
            linewidth=linewidths[i % len(linewidths)]
        )
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)



def plot_attn_bk(attention):
    # Plot attention map
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(attention, cmap='viridis', origin='upper')

    # Add colorbar
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Random Attention Map")

    # Add random rectangle
    x_start = random.randint(0, 90)
    y_start = random.randint(0, 90)
    width = random.randint(5, 10)
    height = random.randint(5, 10)

    rect = plt.Rectangle((x_start, y_start), width, height,
                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.show()


def cst_attn_2d_block_model(adapt_data, out_png="", suptitle=""):
    fig, axs = plt.subplots(4, 2, figsize=(15, 10))
    for i in range(4):
        for j in range(2):
            sub_n = "{}_{}".format(i, j)
            title, attn, x_ltl_set, y_ltl_set, kwargs = adapt_data[sub_n]
            plot_attn_with_rect(axs[i][j], title, attn, x_ltl_set, y_ltl_set, kwargs)
    fig.suptitle(suptitle)
    #fig.tight_layout()
    fig.savefig(out_png, dpi=300)


def cst_melattn_2d_type_model(adapt_data, out_png="", suptitle=""):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(2):
        for j in range(3):
            sub_n = "{}_{}".format(i, j)
            if i == 1 and j == 0:
                axs[i, j].axis('off')  # Blank subplot
            else:
                if sub_n in ["0_1", "0_2"]:
                    plot_attn_with_rect(axs[i][j], *adapt_data[sub_n])
                else:
                    plot_mel_with_pitch(axs[i][j], *adapt_data[sub_n])
    fig.suptitle(suptitle)
    # fig.tight_layout()
    fig.savefig(out_png, dpi=300)

def plot_simple_mel(audio_path, out_path):
    # === Step 1: Load audio ===
    y, sr = librosa.load(audio_path, sr=16000)

    # === Step 2: Compute Mel spectrogram ===
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # === Step 3: Estimate pitch ===
    f0, voiced_flag, voiced_prob = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'),
                                                fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)

    # === Step 5: Plot Mel Spectrogram with Pitch and Word Labels ===
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot spectrogram
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)

    # Plot pitch
    ax.plot(times, f0, color='cyan', linewidth=1, label='Pitch (Hz)')
    ax.legend(loc='upper right')
    ax.set_title('Mel Spectrogram with Pitch and Word Labels')

    # Word-level x-tick labels
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_path)


def plot_f0_comparison(F0_ref, F0_pred, F0_fused, title="Pitch Comparison", out_path="pitch_compare.png"):
    """
    Visualize reference, predicted, and fused F0 contours.
    F0_* can be 1-D torch tensors or numpy arrays.
    """
    # convert to numpy
    F0_ref   = F0_ref.detach().cpu().numpy() if isinstance(F0_ref, torch.Tensor) else F0_ref
    F0_pred  = F0_pred.detach().cpu().numpy() if isinstance(F0_pred, torch.Tensor) else F0_pred
    F0_fused = F0_fused.detach().cpu().numpy() if isinstance(F0_fused, torch.Tensor) else F0_fused

    # match length if needed
    T = len(F0_fused)
    if len(F0_ref)  != T:
        F0_ref  = np.interp(np.linspace(0,1,T), np.linspace(0,1,len(F0_ref)),  F0_ref)
    if len(F0_pred) != T:
        F0_pred = np.interp(np.linspace(0,1,T), np.linspace(0,1,len(F0_pred)), F0_pred)

    plt.figure(figsize=(10, 4))
    plt.plot(F0_pred,  label='Predicted F₀', color='blue',  linestyle='--', alpha=0.8)
    plt.plot(F0_ref,   label='Reference F₀', color='green', linestyle=':',  alpha=0.8)
    plt.plot(F0_fused, label='Fused F₀',     color='red',   linewidth=2.0)
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel("Normalized F₀ (log or latent scale)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)


def plot_f0_multi(
    f0_curves,              # list of arrays/tensors: [curve1, curve2, curve3, ...]
    model_names,            # list of strings:      ["pred", "ref", "fused", ...]
    title="Pitch Comparison",
    out_path="pitch_compare.png",
    target_len=None,
    ref_idx=None,           # index of reference curve (optional)
    need_interpolate=True
):
    """
    Plot multiple F0 curves from different models.

    Args:
        f0_curves   : list of 1-D F0 sequences (torch.Tensor or numpy array)
        model_names : list of legend names (same length as f0_curves)
        ref_idx     : index of the reference F0 for special highlighting
        target_len  : force all curves to this length; if None → use max length
    """

    assert len(f0_curves) == len(model_names), "Length mismatch between curves and names."

    # Convert to numpy
    curves_np = []
    for c in f0_curves:
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        curves_np.append(np.asarray(c))

    # Determine target length
    if target_len is None:
        target_len = max(len(c) for c in curves_np)

    # Interpolate all curves to target length
    curves_resampled = []
    if need_interpolate:
        for c in curves_np:
            if len(c) != target_len:
                c = np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(c)),
                    c
                )
            curves_resampled.append(c)
        curves_np = curves_resampled

    # Plot
    plt.figure(figsize=(12, 5))

    for i, (c, name) in enumerate(zip(curves_np, model_names)):
        if i == ref_idx:
            # Reference curve: highlight style
            plt.plot(c, label=name, linewidth=2.0, linestyle=":", alpha=0.9)
        else:
            plt.plot(c, label=name, linewidth=1.4, alpha=0.85)

    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel("F₀ (Hz or log-scale)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def draw_dtw(a, b, best_path, output_png="dtw_out.png"):
    # -----------------------------
    # Plot contours + alignment
    # -----------------------------
    plt.figure(figsize=(12, 5))

    # plot both contours
    plt.plot(a, label="Series A", color="blue")
    plt.plot(b, label="Series B", color="red")

    # draw alignment lines
    for i, j in best_path:
        plt.plot([i, j], [a[i], b[j]], color="gray", alpha=0.4, linewidth=0.8)

    plt.xlabel("Time index")
    plt.ylabel("Pitch (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="/home/rosen/Project/StableTTS/result/ddpm_dit_cross/0019_001453.wav")
    parser.add_argument("--out", type=str,
                        default="/home/rosen/Project/StableTTS/result/ddpm_dit_cross/mel_out.png")
    args = parser.parse_args()

    plot_simple_mel(args.audio, args.out)
    #audio = "/home/rosen/Project/StableTTS/result/ddpm_dit_cross/0019_001453.wav"
    # === Create 2x2 subplots ===

    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:
                axs[i, j].axis('off')  # Blank subplot
            else:
                plot_mel_with_pitch(axs[i][j], audio)
    plt.tight_layout()
    plt.show()
    """
    
