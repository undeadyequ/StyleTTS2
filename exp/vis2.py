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



def plot_attn_with_rect(ax, title, attn, x_ltl_set, y_ltl_set, kwargs):
    rect_line_width = 0.5
    rect_line_style = "--"
    xlabel, xticks, x_ticklabs = x_ltl_set
    ylabel, yticks, y_ticklabs = y_ltl_set

    ax.set_title(title, fontsize=kwargs["fontsize"])
    pc = ax.pcolor(attn, cmap=plt.cm.Blues, alpha=0.9)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlabel, fontsize=kwargs["fontsize"])
    ax.set_ylabel(ylabel, fontsize=kwargs["fontsize"], rotation=kwargs["yticklabel_rotation"], loc="top")

    if x_ticklabs is not None:
        ax.set_xticklabels(labels=x_ticklabs, fontsize=kwargs["fontsize"], rotation=kwargs["x_rotation"], ha="left")
    else:
        ax.set_xticklabels([], visible=False)
    if y_ticklabs is not None:
        ax.set_yticklabels(labels=y_ticklabs, fontsize=kwargs["fontsize"], rotation=kwargs["y_rotation"], va="center")
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

def plot_mel_with_pitch(ax, title, speech, x_ltl_set, kwargs, show_pitch=False):
    y, sr = librosa.load(speech, sr=None)
    hop_length = 200
    sr = 16000
    n_fft = 1024
    xlabel, xticks, x_ticklabs = x_ltl_set

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis=None, y_axis='mel', sr=sr, hop_length=hop_length, cmap='magma', ax=ax)

    # Estimate pitch (f0)
    if show_pitch:
        pitch, t = pw.dio(
            y.astype(np.float64),
            sr,
            frame_period=hop_length / sr * 1000)
        pitch = pw.stonemask(y.astype(np.float64), pitch, t, sr)
        times = librosa.times_like(pitch, sr=sr, hop_length=hop_length)
        ax.plot(times, pitch, color='cyan', linewidth=1.5, label='Pitch')
        print("speech:{} len of f0 {} and x_ticks {}".format(speech, len(pitch), xticks[-1]))
    ax.set_title(title)
    #ax.set_xticks(xticks)
    #ax.set_xticklabels(labels=x_ticklabs, fontsize=kwargs["fontsize"], rotation=kwargs["x_rotation"], ha="left")
    #ax.set_xticks(xticks)
    #ax.set_xlabel(xlabel, fontsize=kwargs["fontsize"])
    #if x_ticklabs is not None:
    #    ax.set_xticklabels(labels=x_ticklabs, fontsize=kwargs["fontsize"], rotation=kwargs["x_rotation"], ha="left")

    #ax.set_xticks(xticks)
    #ax.set_xlabel(xlabel)

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
    ax.legend(loc='upper right')

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
    ax.legend(loc='upper right')


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
    
