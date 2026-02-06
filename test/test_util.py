from utils import extract_pitch_trend_v2
import torch
import torch.nn.functional as F
import numpy as np

def test_extract_pitch_trend_inference():
    # 1. Setup dimensions
    B = 2
    Tp_gd, Tf_gd = 4, 100  # Reference: 4 phonemes, 100 frames
    Tp_syn, Tf_syn = 8, 150  # Target: 8 phonemes, 150 frames
    tgt_len = 128  # Final BERT embedding length

    # 2. Mock Pitch Ground Truth (Reference Audio)
    # Let's create a simple rising pitch from 100Hz to 200Hz
    pitch_gd_frame = torch.linspace(100, 200, Tf_gd).repeat(B, 1)

    # 3. Mock Ground Truth Attention [B, Tp, Tf]
    # Simple linear alignment: each phoneme gets 25 frames
    attn_f2p_gd = torch.zeros(B, Tp_gd, Tf_gd)
    for i in range(Tp_gd):
        attn_f2p_gd[:, i, i * 25:(i + 1) * 25] = 1.0

    # 4. Mock Predicted Attention [B, Tp_syn, Tf_syn]
    # Target has 8 phonemes, each getting ~18-19 frames
    attn_f2p_pred = torch.zeros(B, Tp_syn, Tf_syn)
    step = Tf_syn // Tp_syn
    for i in range(Tp_syn):
        attn_f2p_pred[:, i, i * step:(i + 1) * step] = 1.0

    print(f"Input Shape: {pitch_gd_frame.shape}")
    print(f"GD Phonemes: {Tp_gd}, Pred Phonemes: {Tp_syn}")

    # 5. Run Function
    try:
        trend = extract_pitch_trend_v2(
            pitch_gd_frame=pitch_gd_frame,
            attn_f2p_gd=attn_f2p_gd,
            tgt_frame_len=tgt_len,
            attn_f2p_pred=attn_f2p_pred,
            fmin=50.0,
            fmax=600.0
        )

        # 6. Validations
        assert trend.shape == (B, 1, tgt_len), f"Wrong output shape: {trend.shape}"
        assert torch.isfinite(trend).all(), "Trend contains NaNs or Infs"

        # Check if the pitch is still in a reasonable log range
        # log(100) approx 4.6, log(200) approx 5.3
        assert trend.mean() > 4.0 and trend.mean() < 6.0, f"Unreasonable log-pitch: {trend.mean()}"

        print("✅ Test Passed: Shape and values are correct.")
        print(f"Output Shape: {trend.shape}")

    except Exception as e:
        print(f"❌ Test Failed: {str(e)}")
        raise e


if __name__ == '__main__':
    test_extract_pitch_trend_inference()