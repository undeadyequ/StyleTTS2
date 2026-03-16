"""Cross-attention map extraction and visualization.

Saves the CFM decoder's cross-attention maps as .npy and draws a PDF grid
using vis_tbh_cross_attention_time_grouped.

Saved attn_maps shape: [T_ode, B_blocks, H_heads, L_mel, L_ref]
  T_ode   : ODE timesteps captured (first=early, last=late selected for plot)
  B_blocks: number of cross-attention blocks in the DiT estimator
  H_heads : attention heads per block (4)
  L_mel   : synthesized mel frame count
  L_ref   : reference prosodic frame count

Usage:
    # Mode A – Synthesize + save wav + attn.npy + draw PDF
    python exp/exec_crossAttn.py \\
        --text "She sells seashells by the seashore." \\
        --ref_wav /path/to/ref.wav \\
        --ref_txt "I know you." \\
        --model_name decodit_cfm_v29 \\
        --out_dir exp/res/cross_attn \\
        --blocks 3 4

    # Mode B – Draw from an existing .npy (skip synthesis)
    python exp/exec_crossAttn.py \\
        --attn_npy exp/res/cross_attn/ref_crossattn.npy \\
        --out_pdf  exp/res/cross_attn/ref_crossattn_attn.pdf \\
        --blocks 3 4
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path

os.chdir('/home/rosen/Project/StyleTTS2')
sys.path.append('/home/rosen/Project/StyleTTS2')


# ── model registry (matches inferenceAPI_bertFusion.py __main__) ──────────────
_MODEL_ROOT = '/home/rosen/ckpt/styletts2_libriTTS/'
_MODEL_CONFIG = {
    # model_name: [ckpt_relpath, config_relpath]
    'mdit_cfm_v10':    ['first_txt2mel_cfm_v10/epoch_2nd_00048.pth',
                        'first_txt2mel_cfm_v10/config_libritts_txt2mel_cfm_v10.yml'],
    'mdit_cfm_v20':    ['first_txt2mel_cfm_v20/epoch_2nd_00018.pth',
                        'first_txt2mel_cfm_v20/config_libritts_txt2mel_cfm_v20.yml'],
    'decodit_cfm_v26': ['first_txt2mel_cfm_v26/epoch_2nd_00030.pth',
                        'first_txt2mel_cfm_v26/config_libritts_txt2mel_cfm_v26.yml'],
    'decodit_cfm_v29': ['first_txt2mel_cfm_v29/epoch_2nd_00048.pth',
                        'first_txt2mel_cfm_v29/config_libritts_txt2mel_cfm_v29.yml'],
}


# ── save / load ───────────────────────────────────────────────────────────────

def save_attn(attn_maps, attn_npy_path: str) -> np.ndarray:
    """Save attn_maps (all ODE steps) as .npy.

    Shape: [T_ode, B_blocks, H_heads, L_mel, L_ref]
    """
    Path(attn_npy_path).parent.mkdir(parents=True, exist_ok=True)
    arr = attn_maps.cpu().numpy() if isinstance(attn_maps, torch.Tensor) else attn_maps
    np.save(attn_npy_path, arr)
    print(f"  Saved attn maps {arr.shape} → {attn_npy_path}")
    return arr


# ── visualization ─────────────────────────────────────────────────────────────

def draw_attn(
    attn_npy_path: str,
    out_pdf: str,
    blocks=(0, 5),
    block_labels=None,
    time_labels=None,
    figsize=(36, 12),
):
    """Load saved .npy and draw cross-attention PDF.

    Selects the first (early diffusion) and last (late diffusion) ODE
    timesteps and the two block indices in `blocks`, then calls
    vis_tbh_cross_attention_time_grouped which expects (T=2, B=2, H=4, L, L).

    Args:
        attn_npy_path: Path to .npy saved by save_attn().
        out_pdf:       Output PDF path.
        blocks:        Pair of DiT block indices to show (e.g. (3, 4)).
        block_labels:  Row label overrides (list of 2 strings).
        time_labels:   Column group header overrides (list of 2 strings).
        figsize:       Figure size in inches.
    """
    from exp.visualization import vis_tbh_cross_attention_time_grouped

    attn_all = np.load(attn_npy_path, allow_pickle=True)
    T, B_all, H, Lm, Lr = attn_all.shape
    print(f"  Loaded attn_maps: T={T}, B={B_all}, H={H}, L_mel={Lm}, L_ref={Lr}")

    b0, b1 = int(blocks[0]), int(blocks[1])
    assert b0 < B_all and b1 < B_all, (
        f"Block indices {b0},{b1} out of range (model has {B_all} cross-attn blocks)")

    # Select [first, last] ODE step × [b0, b1] blocks × 4 heads → (2, 2, 4, L, L)
    tbh_sel = attn_all[np.ix_([0, T - 1], [b0, b1], [0, 1, 2, 3])]

    if block_labels is None:
        block_labels = [f"Block {b0 + 1}", f"Block {b1 + 1}"]
    if time_labels is None:
        time_labels = [r"Early diffusion ($t \approx 0$)",
                       r"Late diffusion ($t \approx T$)"]

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    vis_tbh_cross_attention_time_grouped(
        tbh_sel,
        time_labels=time_labels,
        block_labels=block_labels,
        head_labels=[f"Head {i + 1}" for i in range(4)],
        title=None,
        dashed_time_separator=False,
        savepath=out_pdf,
        figsize=figsize,
    )
    print(f"  Saved cross-attn plot → {out_pdf}")


# ── full pipeline ─────────────────────────────────────────────────────────────

def run(
    text: str,
    ref_wav: str,
    ref_txt: str,
    model_name: str,
    model_ckpt: str,
    model_config: str,
    out_dir: str,
    alpha: float = 0.3,
    beta: float = 0.7,
    cfg_strength: float = 3.0,
    blocks: tuple = (0, 5),
    figsize: tuple = (36, 12),
    speech_id: str = None,
):
    """Full pipeline: synthesize → save wav + attn.npy → draw PDF.

    Uses InferenceAPI.synthesize_one(return_attn_map=True) so no inference
    logic is duplicated here.

    Args:
        text:         Text to synthesize.
        ref_wav:      Reference wav path (style source).
        ref_txt:      Transcript of the reference wav (required by InferenceAPI).
        model_name:   Model identifier (e.g. 'decodit_cfm_v29').
        model_ckpt:   Path to model checkpoint (.pth).
        model_config: Path to model config (.yml).
        out_dir:      Output directory (created if missing).
        blocks:       Two DiT block indices to show in the attn plot.
        speech_id:    Output filename prefix (default: ref wav basename).

    Returns:
        (wav_path, attn_npy_path)
    """
    from exp.inferenceAPI_bertFusion import InferenceAPI

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"Cross-Attention Extraction")
    print(f"  Text    : {text!r}")
    print(f"  Ref wav : {ref_wav}")
    print(f"  Model   : {model_name}")
    print(f"  Blocks  : {blocks}")
    print(f"  Out dir : {out_dir}")
    print(f"{'='*60}\n")

    print("Loading model...")
    api = InferenceAPI(
        model_name=model_name,
        ckpt_path=model_ckpt,
        config_path=model_config,
        device=device,
    )

    print("Synthesizing with return_attn_map=True...")
    audio, attn_maps = api.synthesize_one(
        text=text,
        ref_wav_path=ref_wav,
        ref_txt=ref_txt,
        alpha=alpha,
        beta=beta,
        cfg_strength=cfg_strength,
        return_attn_map=True,
    )

    if speech_id is None:
        speech_id = os.path.splitext(os.path.basename(ref_wav))[0] + '_crossattn'

    # Save synthesized wav
    wav_path = os.path.join(out_dir, f"{speech_id}.wav")
    torchaudio.save(wav_path, torch.from_numpy(audio).unsqueeze(0), 24000)
    print(f"  Saved wav → {wav_path}")

    attn_npy_path = None
    if attn_maps is not None:
        attn_npy_path = os.path.join(out_dir, f"{speech_id}.npy")
        save_attn(attn_maps, attn_npy_path)

        out_pdf = os.path.join(out_dir, f"{speech_id}_attn.pdf")
        draw_attn(attn_npy_path, out_pdf, blocks=blocks, figsize=figsize)
    else:
        print("  ⚠ attn_maps is None — model may not have cross-attention blocks")

    print(f"\nDone. Outputs in: {out_dir}")
    return wav_path, attn_npy_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract and visualize cross-attention maps from the mdit_cfm/decodit decoder.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode A: synthesize + draw
    synth_g = parser.add_argument_group('Mode A – synthesize + draw')
    synth_g.add_argument('--text',         type=str, help='Text to synthesize')
    synth_g.add_argument('--ref_wav',      type=str, help='Reference wav path')
    synth_g.add_argument('--ref_txt',      type=str, default='',
                         help='Transcript of the reference wav')
    synth_g.add_argument('--model_name',   type=str, default='decodit_cfm_v29',
                         help='Model name key (default: decodit_cfm_v29)')
    synth_g.add_argument('--model_ckpt',   type=str, default=None,
                         help='Checkpoint path (auto-resolved from --model_name if omitted)')
    synth_g.add_argument('--model_config', type=str, default=None,
                         help='Config path (auto-resolved from --model_name if omitted)')
    synth_g.add_argument('--out_dir',      type=str, default='exp/res/cross_attn')
    synth_g.add_argument('--alpha',        type=float, default=0.3)
    synth_g.add_argument('--beta',         type=float, default=0.7)
    synth_g.add_argument('--cfg',          type=float, default=3.0, dest='cfg_strength')
    synth_g.add_argument('--speech_id',    type=str, default=None,
                         help='Output filename prefix (default: ref wav basename)')

    # Mode B: draw only
    draw_g = parser.add_argument_group('Mode B – draw from existing .npy')
    draw_g.add_argument('--attn_npy', type=str, default=None,
                        help='Path to existing .npy (skips synthesis)')
    draw_g.add_argument('--out_pdf',  type=str, default=None,
                        help='Output PDF path (default: <attn_npy stem>_attn.pdf)')

    # Shared
    parser.add_argument('--blocks', type=int, nargs=2, default=[0, 5],
                        metavar=('B0', 'B1'),
                        help='Two DiT block indices to visualize (default: 0 5)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[36, 12],
                        metavar=('W', 'H'), help='Figure size in inches (default: 36 12)')

    args = parser.parse_args()

    if args.attn_npy is not None:
        # Mode B: draw only
        out_pdf = args.out_pdf or args.attn_npy.replace('.npy', '_attn.pdf')
        draw_attn(
            attn_npy_path=args.attn_npy,
            out_pdf=out_pdf,
            blocks=tuple(args.blocks),
            figsize=tuple(args.figsize),
        )

    elif args.text and args.ref_wav:
        # Resolve ckpt / config from registry if not given explicitly
        ckpt   = args.model_ckpt
        config = args.model_config
        if ckpt is None or config is None:
            if args.model_name not in _MODEL_CONFIG:
                raise ValueError(
                    f"Unknown model_name {args.model_name!r}. "
                    f"Provide --model_ckpt / --model_config, or add to _MODEL_CONFIG. "
                    f"Known: {list(_MODEL_CONFIG)}")
            rel_ckpt, rel_cfg = _MODEL_CONFIG[args.model_name]
            ckpt   = ckpt   or os.path.join(_MODEL_ROOT, rel_ckpt)
            config = config or os.path.join(_MODEL_ROOT, rel_cfg)

        # Read ref_txt from .lab if not provided
        ref_txt = args.ref_txt
        if not ref_txt:
            lab = os.path.splitext(args.ref_wav)[0] + '.lab'
            if os.path.exists(lab):
                with open(lab) as f:
                    ref_txt = f.read().strip()
                print(f"  ref_txt loaded from {lab}: {ref_txt!r}")
            else:
                ref_txt = args.text   # fallback: use synthesis text
                print(f"  ref_txt not provided and no .lab found; using synthesis text as fallback")

        run(
            text=args.text,
            ref_wav=args.ref_wav,
            ref_txt=ref_txt,
            model_name=args.model_name,
            model_ckpt=ckpt,
            model_config=config,
            out_dir=args.out_dir,
            alpha=args.alpha,
            beta=args.beta,
            cfg_strength=args.cfg_strength,
            blocks=tuple(args.blocks),
            figsize=tuple(args.figsize),
            speech_id=args.speech_id,
        )

    else:
        parser.print_help()
        sys.exit(1)
